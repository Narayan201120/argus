"""P4-1 parallel dispatch engine + opening-round runner (DEC-053, backend only, mock-only).

Round zero fans out the baseline tools concurrently, races them against the
per-investigation cancel event and the wall-clock deadline, then folds the
surviving results onto the evidence board with source_ref dedupe.

Background-runner contract: run_opening_round never raises; every storage or
transition failure is logged and the runner returns.
"""

import asyncio
import time
from typing import Any

from app.config import settings
from app.evidence.models import Evidence, InvestigationStatus, StatusReason
from app.investigations import TERMINAL, manager, new_evidence_id
from app.metrics import FIRST_EVIDENCE_LATENCY, TOOL_CALLS, TOOL_LATENCY
from app.tools.base import BaseTool, ToolResult
from app.tools.registry import build_tool_registry
from app.utils.logger import get_logger

logger = get_logger(__name__)

BASELINE_ROUND_ZERO: tuple[str, ...] = ("radar_search", "rag_retrieve")


async def _run_one(tool: BaseTool, query: str) -> ToolResult:
    """Run one tool under the shared tool timeout; every failure becomes ok=False."""
    start = time.monotonic()
    try:
        return await asyncio.wait_for(tool.run(query), settings.tool_timeout_s)
    except TimeoutError:
        elapsed_ms = max(int((time.monotonic() - start) * 1000), 0)
        return ToolResult(
            tool_name=tool.name, ok=False, items=[], error="timeout", latency_ms=elapsed_ms
        )
    except Exception as exc:  # noqa: BLE001 - the dispatcher folds every tool failure into metrics
        elapsed_ms = max(int((time.monotonic() - start) * 1000), 0)
        return ToolResult(
            tool_name=tool.name,
            ok=False,
            items=[],
            error=str(exc) or type(exc).__name__,
            latency_ms=elapsed_ms,
        )


async def _race_tools(
    investigation_id: str,
    tools: list[BaseTool],
    query: str,
    deadline_at: float,
) -> dict[str, ToolResult] | None:
    """Run reserved tools concurrently against the cancel event and wall clock.

    Returns {tool_name: result} once every tool finishes. Returns None when the
    cancel event fires or the wall-clock deadline passes (pending work is
    cancelled and suppressed; the terminal state is owned elsewhere).
    """
    event = manager.cancel_event(investigation_id)
    pending: dict[asyncio.Task[ToolResult], BaseTool] = {
        asyncio.ensure_future(_run_one(tool, query)): tool for tool in tools
    }
    watcher: asyncio.Task[bool] = asyncio.ensure_future(event.wait())
    results: dict[str, ToolResult] = {}
    try:
        while pending:
            remaining = max(0.0, deadline_at - time.time())
            wait_for: set[asyncio.Task[Any]] = set(pending)
            wait_for.add(watcher)
            done, _ = await asyncio.wait(
                wait_for, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done or event.is_set():
                return None
            if not done:
                # Deadline passed but the supervisor has not fired yet; bail out
                # and let it record the wall-clock expiry.
                return None
            for task in done:
                tool = pending.pop(task)
                results[tool.name] = task.result()
        return results
    finally:
        if not watcher.done():
            watcher.cancel()
        for task in pending:
            task.cancel()
        leftovers: list[asyncio.Task[Any]] = [watcher, *pending]
        await asyncio.gather(*leftovers, return_exceptions=True)


async def run_opening_round(investigation_id: str) -> None:
    """Run baseline round zero for one investigation. Never raises."""
    try:
        await _run_opening_round(investigation_id)
    except Exception as exc:  # noqa: BLE001 - background runner must never raise
        logger.warning(
            {
                "message": "Opening round failed",
                "investigation_id": investigation_id,
                "error": str(exc),
            }
        )


async def _run_opening_round(investigation_id: str) -> None:
    inv = await manager.get(investigation_id)
    if inv is None:
        logger.warning(
            {"message": "Opening round: investigation not found", "investigation_id": investigation_id}
        )
        return
    if inv.status in TERMINAL:
        return

    if inv.status != InvestigationStatus.GATHERING:
        try:
            moved = await manager.transition(investigation_id, InvestigationStatus.GATHERING)
        except ValueError as exc:
            logger.warning(
                {
                    "message": "Opening round: cannot enter GATHERING",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )
            return
        if moved is None:
            logger.warning(
                {"message": "Opening round: investigation vanished", "investigation_id": investigation_id}
            )
            return
        inv = moved

    query = inv.query
    deadline_at = inv.deadline_at
    created_at = inv.created_at
    pre_existing_refs = {e.source_ref for e in inv.board.evidence}
    pre_existing_count = len(inv.board.evidence)

    registry = build_tool_registry()

    reserved: list[BaseTool] = []
    for name in BASELINE_ROUND_ZERO:
        tool = registry.get(name)
        if tool is None:
            logger.warning(
                {
                    "message": "Opening round: baseline tool missing from registry",
                    "investigation_id": investigation_id,
                    "tool": name,
                }
            )
            TOOL_CALLS.labels(tool=name, status="missing").inc()
            continue
        if not tool.enabled:
            TOOL_CALLS.labels(tool=tool.name, status="disabled").inc()
            continue
        row = await manager.record_tool_call(investigation_id)
        if row is None:
            logger.warning(
                {
                    "message": "Opening round: investigation vanished during budget reserve",
                    "investigation_id": investigation_id,
                }
            )
            return
        if row.status in TERMINAL or row.status_reason is not None:
            # Budget exhausted (or a raced cancel/expiry ended the run); the
            # recorded terminal state stands, so stop dispatching.
            return
        reserved.append(tool)

    results: dict[str, ToolResult] = {}
    if reserved:
        raced = await _race_tools(investigation_id, reserved, query, deadline_at)
        if raced is None:
            return
        results = raced

    if manager.cancel_event(investigation_id).is_set():
        return

    seen = set(pre_existing_refs)
    written = 0
    for tool in reserved:
        result = results.get(tool.name)
        if result is None:
            continue  # Every reserved tool produces a result; stay total regardless.
        status = "ok" if result.ok else ("timeout" if result.error == "timeout" else "error")
        TOOL_CALLS.labels(tool=tool.name, status=status).inc()
        TOOL_LATENCY.labels(tool=tool.name).observe(result.latency_ms / 1000)
        if not result.ok:
            continue
        for item in result.items:
            if item.source_ref in seen:
                TOOL_CALLS.labels(tool=tool.name, status="deduped").inc()
                continue
            seen.add(item.source_ref)
            now = time.time()
            evidence = Evidence(
                id=new_evidence_id(),
                investigation_id=investigation_id,
                source_ref=item.source_ref,
                content=item.content,
                type=item.type,
                confidence=item.confidence,
                provenance=dict(item.provenance),
                created_at=now,
            )
            try:
                await manager.add_evidence(investigation_id, evidence)
            except ValueError as exc:
                logger.warning(
                    {
                        "message": "Opening round: board closed mid-write",
                        "investigation_id": investigation_id,
                        "tool": tool.name,
                        "error": str(exc),
                    }
                )
                return
            written += 1
            if pre_existing_count == 0 and written == 1:
                FIRST_EVIDENCE_LATENCY.observe(now - created_at)

    if written > 0 or pre_existing_count > 0:
        return  # Park in GATHERING with evidence on the board.
    if not reserved:
        return  # Nothing was ever enabled/attempted; park in GATHERING with an empty board.
    try:
        await manager.transition(
            investigation_id, InvestigationStatus.FAILED, StatusReason.PROVIDER_FAILURE
        )
    except ValueError as exc:
        logger.warning(
            {
                "message": "Opening round: cannot mark PROVIDER_FAILURE",
                "investigation_id": investigation_id,
                "error": str(exc),
            }
        )
    return
