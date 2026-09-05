"""P4-3 milestone synthesizer worker (DEC-053, backend only).

Calls a real LLM connector with streaming deltas and persists Markdown
milestone reports. Tests monkeypatch the connectors, never the network.

Note: connector pick + one-step failover intentionally duplicates the small
selection helper from app/analysis/workers.py so synthesis stays independent
of the analysis/critique/gap worker path.
"""

import inspect
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field, TypeAdapter

import app.investigations as investigations_module
from app.analysis import board as board_module
from app.analysis import events as events_module
from app.config import settings
from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.connectors.registry import registry
from app.costs import estimate_llm_cost
from app.metrics import (
    SYNTHESIS_LATENCY,
    SYNTHESIS_TOTAL,
    TIME_TO_USEFUL_ANSWER,
    record_role_tokens,
)
from app.rediskit import holder
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYNTHESIS_WORKER = "synthesis"

SYNTHESIS_PROMPT = (
    "You are the ARGUS milestone synthesis worker. Write a faithful Markdown report "
    "from the investigation query and evidence board. "
    "Include sections for findings, contested claims (mark with caution), "
    "and gaps/missing evidence. "
    "Declare limits plainly. Never invent sources or cite evidence not shown on the board."
)

SYNTHESIS_CONFIG = ConnectorConfig(temperature=0.2, max_tokens=4000)


class SynthesisError(Exception):
    """Synthesis failure as data; the investigation loop catches this."""


class SynthesisRecord(BaseModel):
    milestone: int = Field(ge=0)
    markdown: str = Field(min_length=1, max_length=60000)
    final: bool = False
    created_at: float


def _pick_connector() -> BaseConnector:
    """Select the connector for milestone synthesis."""
    pinned = settings.synthesis_connector_id.strip()
    if pinned:
        connector = registry.get(pinned)
        if connector is None:
            raise SynthesisError(f"provider_error: unknown synthesis connector {pinned!r}")
        if not connector.is_available:
            raise SynthesisError(f"provider_error: synthesis connector {pinned!r} unavailable")
        return connector
    available = registry.available()
    if not available:
        raise SynthesisError("provider_error: no available connectors")
    return available[0]


def _failover_candidate(exclude_id: str) -> BaseConnector | None:
    """Next available connector other than the one that just failed."""
    for connector in registry.available():
        if connector.connector_id != exclude_id:
            return connector
    return None


def _provider_error(response: ConnectorResponse) -> SynthesisError:
    detail = response.error or response.status.value
    return SynthesisError(f"provider_error: {detail}")


def _report_key(investigation_id: str) -> str:
    return f"argus:inv:{investigation_id}:report"


def _as_str(raw: str | bytes) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def _accepts_investigation_id(fn: object) -> bool:
    """True when fn takes an investigation_id keyword (or **kwargs).

    Keeps 2-arg test fakes working: run_milestone passes the id only when
    the (possibly monkeypatched) synthesize_board accepts it.
    """
    try:
        params = inspect.signature(fn).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if "investigation_id" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


async def _call_synthesize_board(
    board_text: str,
    query: str,
    milestone: int,
    emit: Callable[[str], Awaitable[None]] | None,
    investigation_id: str,
) -> str:
    target = globals().get("synthesize_board", synthesize_board)
    if _accepts_investigation_id(target):
        return await target(
            board_text, query, milestone, emit=emit, investigation_id=investigation_id
        )
    return await target(board_text, query, milestone, emit=emit)


_records_adapter: TypeAdapter[list[SynthesisRecord]] = TypeAdapter(list[SynthesisRecord])


class SynthesisStore:
    """In-memory authoritative report mirror with best-effort Redis snapshot."""

    def __init__(self) -> None:
        self._memory: dict[str, list[SynthesisRecord]] = {}

    async def save_all(
        self, investigation_id: str, records: list[SynthesisRecord], ttl_s: int
    ) -> None:
        """Mirror in memory, then best-effort Redis snapshot. Never raises."""
        try:
            self._memory[investigation_id] = list(records)
        except Exception as exc:  # noqa: BLE001 - fail open, always
            logger.warning(
                {
                    "message": "SynthesisStore save failed (ignored)",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )
            return
        client = holder.client
        if client is None:
            return
        try:
            payload = _records_adapter.dump_json(records).decode("utf-8")
            await client.set(_report_key(investigation_id), payload)
            await client.expire(_report_key(investigation_id), ttl_s)
        except Exception as exc:  # noqa: BLE001 - fail open, always
            logger.warning(
                {
                    "message": "SynthesisStore snapshot failed (ignored)",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )

    async def load(self, investigation_id: str) -> list[SynthesisRecord]:
        """Memory hit returns immediately; else reassemble from Redis. Never raises."""
        try:
            cached = self._memory.get(investigation_id)
            if cached is not None:
                return list(cached)
            client = holder.client
            if client is None:
                return []
            try:
                raw = await client.get(_report_key(investigation_id))
            except Exception as exc:  # noqa: BLE001 - fail open, always
                logger.warning(
                    {
                        "message": "SynthesisStore load failed (ignored)",
                        "investigation_id": investigation_id,
                        "error": str(exc),
                    }
                )
                return []
            if not raw:
                return []
            try:
                records = _records_adapter.validate_json(_as_str(raw))
            except Exception as exc:  # noqa: BLE001 - corrupt snapshot reads as empty
                logger.warning(
                    {
                        "message": "SynthesisStore snapshot parse failed (ignored)",
                        "investigation_id": investigation_id,
                        "error": str(exc),
                    }
                )
                return []
            self._memory[investigation_id] = list(records)
            return list(records)
        except Exception as exc:  # noqa: BLE001 - load never raises
            logger.warning(
                {
                    "message": "SynthesisStore load failed (ignored)",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )
            return []


synthesis_store = SynthesisStore()


def _is_ok(response: ConnectorResponse) -> bool:
    return response.status == ConnectorStatus.SUCCESS and bool(
        response.content and response.content.strip()
    )


async def synthesize_board(
    board_text: str,
    query: str,
    milestone: int,
    emit: Callable[[str], Awaitable[None]] | None = None,
    *,
    investigation_id: str | None = None,
) -> str:
    """Render one milestone Markdown report, streaming deltas via emit.

    Prefers connector.stream_query chunks (AsyncIterator[str] per
    app/connectors/base.py); falls back to single-shot query. Emits each
    non-empty delta; generic emit failures are swallowed so a slow live
    tail cannot fail synthesis, but SynthesisError from emit (cancel abort)
    propagates to stop cleanly.
    """
    start = time.perf_counter()
    status = "ok"
    try:
        task = (
            f"{SYNTHESIS_PROMPT}\n\n"
            f"Original investigation query: {query}\n\n"
            f"Milestone: {milestone}\n\n"
            f"Evidence board:\n{board_text}\n\n"
            "Write the milestone report in Markdown."
        )
        config = SYNTHESIS_CONFIG
        primary = _pick_connector()
        fallback = _failover_candidate(primary.connector_id)

        async def _drain_stream(connector: BaseConnector) -> list[str]:
            chunks: list[str] = []
            stream = getattr(connector, "stream_query", None)
            if stream is None:
                return []
            async for chunk in stream(query, task, config):
                if not chunk:
                    continue
                chunks.append(chunk)
                if emit is not None:
                    try:
                        await emit(chunk)
                    except SynthesisError:
                        raise
                    except Exception:
                        continue
            return chunks

        parts: list[str] = []
        used_id: str = primary.connector_id
        stream_error: Exception | None = None
        try:
            parts = await _drain_stream(primary)
        except SynthesisError:
            raise
        except Exception as exc:
            stream_error = exc
            parts = []
        if not parts and fallback is not None:
            try:
                fb_parts = await _drain_stream(fallback)
                if fb_parts:
                    parts = fb_parts
                    used_id = fallback.connector_id
                    stream_error = None
            except SynthesisError:
                raise
            except Exception as exc:
                if stream_error is None:
                    stream_error = exc

        if parts:
            full = "".join(parts)
            if not full.strip():
                raise SynthesisError("provider_error: empty synthesis")
            usage: TokenUsage | None = None
            record_role_tokens(SYNTHESIS_WORKER, used_id, usage)
            return full

        # No streamed output: single-shot query with one-step failover.
        response: ConnectorResponse
        used: BaseConnector
        try:
            response = await primary.query(query, task, config)
        except Exception as exc:
            if fallback is None:
                raise SynthesisError(f"provider_error: {exc}") from exc
            try:
                retry = await fallback.query(query, task, config)
            except Exception as exc2:
                raise SynthesisError(f"provider_error: {exc2}") from exc2
            if not _is_ok(retry):
                raise _provider_error(retry) from None
            response, used = retry, fallback
        else:
            if _is_ok(response):
                used = primary
            else:
                if fallback is None:
                    raise _provider_error(response)
                try:
                    retry = await fallback.query(query, task, config)
                except Exception as exc:
                    raise SynthesisError(f"provider_error: {exc}") from exc
                if not _is_ok(retry):
                    raise _provider_error(retry) from None
                response, used = retry, fallback

        full = response.content
        if not full or not full.strip():
            raise SynthesisError("provider_error: empty synthesis")
        if emit is not None:
            try:
                await emit(full)
            except SynthesisError:
                raise
            except Exception:
                pass
        record_role_tokens(SYNTHESIS_WORKER, used.connector_id, response.token_usage)
        if investigation_id and response.token_usage is not None:
            amount = estimate_llm_cost(
                used.connector_id,
                response.token_usage.prompt_tokens,
                response.token_usage.completion_tokens,
            )
            await investigations_module.manager.add_cost(investigation_id, amount)
        return full
    except SynthesisError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise SynthesisError(f"provider_error: {exc}") from exc
    finally:
        SYNTHESIS_LATENCY.observe(time.perf_counter() - start)
        SYNTHESIS_TOTAL.labels(status=status).inc()


async def run_milestone(investigation_id: str, milestone: int, final: bool) -> str | None:
    """Synthesize one milestone report for an investigation. Never raises."""
    try:
        inv = await investigations_module.manager.get(investigation_id)
        if inv is None:
            return None
        if inv.status in investigations_module.TERMINAL:
            return None

        await events_module.publish(
            investigation_id,
            events_module.SYNTHESIS_START,
            {"milestone": milestone, "final": final},
        )

        board_text = board_module.format_board(inv)
        query = inv.query

        async def _emit(delta: str) -> None:
            if investigations_module.manager.cancel_event(investigation_id).is_set():
                raise SynthesisError("cancelled")
            await events_module.publish(
                investigation_id,
                events_module.SYNTHESIS_TOKEN,
                {"milestone": milestone, "delta": delta},
            )

        try:
            markdown = await _call_synthesize_board(
                board_text, query, milestone, _emit, investigation_id
            )
        except SynthesisError as exc:
            logger.warning(
                {
                    "message": "Milestone synthesis failed",
                    "investigation_id": investigation_id,
                    "milestone": milestone,
                    "error": str(exc),
                }
            )
            return None

        record = SynthesisRecord(
            milestone=milestone, markdown=markdown, final=final, created_at=time.time()
        )
        try:
            existing = await synthesis_store.load(investigation_id)
        except Exception:
            existing = []
        existing.append(record)
        try:
            await synthesis_store.save_all(
                investigation_id, existing, settings.investigation_ttl_s
            )
        except Exception as exc:  # noqa: BLE001 - store is fail-open; belt and braces
            logger.warning(
                {
                    "message": "Milestone report save failed (ignored)",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )

        await events_module.publish(
            investigation_id,
            events_module.SYNTHESIS_END,
            {"milestone": milestone, "final": final},
        )
        if milestone == 0:
            try:
                TIME_TO_USEFUL_ANSWER.observe(time.time() - inv.created_at)
            except Exception as exc:  # noqa: BLE001 - metrics must not fail the run
                logger.warning(
                    {
                        "message": "Time-to-useful-answer metric failed (ignored)",
                        "investigation_id": investigation_id,
                        "error": str(exc),
                    }
                )
        return markdown
    except Exception as exc:  # noqa: BLE001 - background runner must never raise
        logger.warning(
            {
                "message": "Milestone synthesis failed",
                "investigation_id": investigation_id,
                "milestone": milestone,
                "error": str(exc),
            }
        )
        return None
