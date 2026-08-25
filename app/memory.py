"""Working-memory session store (Redis-backed, fail-open).

Stores the last N question/answer pairs per ``session_id`` so follow-up
questions can reference earlier context. This is WORKING memory, not
long-term memory: entries expire (MEMORY_TTL_S, default 24h) and the
window rolls (MEMORY_MAX_TURNS). Cross-session recall is out of scope.

Every operation silently no-ops when Redis is unavailable or memory is
disabled - a memory outage can never fail a request.
"""

import json
import time
from typing import Any

from app.config import settings
from app.rediskit import holder
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _key(session_id: str) -> str:
    return f"argus:sess:{session_id}"


class SessionStore:
    async def append(self, session_id: str, question: str, answer: str) -> None:
        """Store one exchange, rolling off turns beyond MEMORY_MAX_TURNS."""
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
        try:
            raw = await client.get(_key(session_id))
            turns: list[dict[str, Any]] = json.loads(raw) if raw else []
            turns.append({"q": question, "a": answer, "ts": time.time()})
            turns = turns[-max(settings.memory_max_turns, 1):]
            await client.set(_key(session_id), json.dumps(turns))
            await client.expire(_key(session_id), max(settings.memory_ttl_s, 60))
        except Exception as exc:  # noqa: BLE001 - fail open, always
            logger.warning({"message": "Memory append failed (ignored)", "error": str(exc)})

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
        deleted = int(await client.delete(_key(session_id)) or 0)
        return deleted > 0


def format_history(turns: list[dict[str, Any]]) -> str | None:
    """Render turns into a bounded transcript for prompt injection.

    Newest exchanges are kept preferentially when the character budget
    (MEMORY_CHAR_CAP) would be exceeded. Returns None when empty.
    """
    if not turns:
        return None
    kept: list[str] = []
    used = 0
    for turn in reversed(turns):
        block = f"User: {turn.get('q', '')}\nAssistant: {turn.get('a', '')}"
        if used + len(block) > settings.memory_char_cap:
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
