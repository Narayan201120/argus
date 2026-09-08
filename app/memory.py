"""Working-memory session store (Redis-backed, fail-open).

Stores the last N question/answer pairs per ``session_id`` so follow-up
questions can reference earlier context. This is WORKING memory, not
long-term memory: entries expire (MEMORY_TTL_S, default 24h) and the
window rolls (MEMORY_MAX_TURNS). Cross-session recall is out of scope.

Every operation silently no-ops when Redis is unavailable or memory is
disabled - a memory outage can never fail a request.
"""

import asyncio
import json
import random
import time
from typing import Any

from redis.exceptions import WatchError

from app.config import settings
from app.metrics import MEMORY_TRUNCATED_ANSWERS
from app.rediskit import holder
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Bounded optimistic-locking retries for SessionStore.append (P6-1).
# fakeredis 2.x has no EVAL support (verified: "unknown command 'eval'"),
# so the atomic append uses WATCH/MULTI instead of a Lua script. This is
# the same code path in prod and in mock-only tests: EXEC fails when
# another writer touched the key between our WATCH and EXEC, and we
# re-read + retry instead of silently losing turns. The backoff is jittered
# so lockstep writers desynchronize instead of colliding every round.
_APPEND_WATCH_RETRIES = 10
_APPEND_RETRY_BACKOFF_S = 0.005


def _key(session_id: str) -> str:
    return f"argus:sess:{session_id}"


class SessionStore:
    async def append(self, session_id: str, question: str, answer: str) -> None:
        """Store one exchange, rolling off turns beyond MEMORY_MAX_TURNS.

        Answers longer than MEMORY_MAX_ANSWER_CHARS are stored truncated:
        the user still sees the full answer; memory keeps its opening so
        one giant reply cannot evict the rest of the conversation.
        """
        if (
            not settings.memory_enabled
            or not session_id
            or not question
            or not answer
        ):
            return
        client = holder.client
        if client is None:
            return
        stored_answer = answer
        if len(stored_answer) > settings.memory_max_answer_chars:
            stored_answer = stored_answer[: settings.memory_max_answer_chars]
            MEMORY_TRUNCATED_ANSWERS.inc()
        new_turn = {"q": question, "a": stored_answer, "ts": time.time()}
        key = _key(session_id)
        max_turns = max(settings.memory_max_turns, 1)
        ttl = max(settings.memory_ttl_s, 60)
        for attempt in range(_APPEND_WATCH_RETRIES):
            try:
                async with client.pipeline(transaction=True) as pipe:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    turns: list[dict[str, Any]] = json.loads(raw) if raw else []
                    turns.append(new_turn)
                    turns = turns[-max_turns:]
                    pipe.multi()
                    pipe.set(key, json.dumps(turns))
                    pipe.expire(key, ttl)
                    await pipe.execute()
                return
            except WatchError:
                # Lost the race: jitter so lockstep writers spread out,
                # then re-read fresh state and retry.
                await asyncio.sleep(random.uniform(0, _APPEND_RETRY_BACKOFF_S * (attempt + 1)))
                continue
            except Exception as exc:  # noqa: BLE001 - fail open, always
                logger.warning({"message": "Memory append failed (ignored)", "error": str(exc)})
                return
        logger.warning({"message": "Memory append failed (ignored)", "error": "watch retries exhausted"})

    async def recent(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Most recent turns, oldest first."""
        if not settings.memory_enabled or not session_id:
            return []
        client = holder.client
        if client is None:
            return []
        try:
            raw = await client.get(_key(session_id))
            turns = json.loads(raw) if raw else []
            limit = limit or settings.memory_inject_turns
            return turns[-max(limit, 1):]
        except Exception as exc:  # noqa: BLE001
            logger.warning({"message": "Memory read failed (ignored)", "error": str(exc)})
            return []

    async def clear(self, session_id: str) -> bool:
        if not settings.memory_enabled or not session_id:
            return False
        client = holder.client
        if client is None:
            return False
        try:
            deleted = int(await client.delete(_key(session_id)) or 0)
            return deleted > 0
        except Exception as exc:  # noqa: BLE001 - fail open like append/recent
            logger.warning({"message": "Memory clear failed (ignored)", "error": str(exc)})
            return False


def format_history(turns: list[dict[str, Any]]) -> str | None:
    """Render turns into a bounded transcript for prompt injection.

    The budget is expressed in tokens (MEMORY_TOKEN_BUDGET) and
    approximated as x4 characters. Newest exchanges are kept
    preferentially when the budget would be exceeded.
    Returns None when empty.
    """
    if not turns:
        return None
    char_budget = max(settings.memory_token_budget, 1) * 4
    kept: list[str] = []
    used = 0
    for turn in reversed(turns):
        block = f"User: {turn.get('q', '')}\nAssistant: {turn.get('a', '')}"
        if used + len(block) > char_budget:
            break
        kept.insert(0, block)
        used += len(block)
    if not kept:
        return None
    return "Earlier conversation (most recent last):\n" + "\n\n".join(kept)


session_store = SessionStore()


async def load_history_text(session_id: str | None) -> str | None:
    """Convenience wrapper used by the API routes."""
    if not session_id:
        return None
    return format_history(await session_store.recent(session_id))
