"""P4-2 adaptive investigation loop (DEC-053, backend only, mock-only).

Drives one investigation from PLANNED through repeated gather/analyze/
critique/gap rounds. Round zero reuses the generic dispatch round runner;
later rounds plan radar/rag follow-ups from the gap worker.

Background-runner contract: run_investigation_loop never raises; every
storage, transition, or worker failure is logged and the runner returns.

Locked decisions (do not change without a new DEC):
- Parking leaves status_reason None. The loop parks in GATHERING whenever
  it stops without a terminal transition; only the manager (budget/supervisor
  expiry), cancel(), or an explicit FAILED transition sets status_reason.
- LOOP_STOPS counting avoids double counting: budget trips are recorded by
  the manager at record time, cancels by cancel(), wall-clock expiry by the
  supervisor. The loop therefore increments LOOP_STOPS only for its own
  endings (sufficient, gap_error, provider_failure) and returns silently for
  budget/cancel/terminal detections owned elsewhere.
- A tool round counts as provider failure only when at least one tool was
  reserved yet nothing was written (written == 0 and attempted). An empty
  reservation (all tools disabled/missing/skipped) parks instead of failing.
- Analysis/critique WorkerErrors are transient: the iteration is skipped and
  retried until wall-clock/budget ends it. A gap WorkerError is fatal for the
  loop (gap_error stop) because without a gap verdict the loop can neither
  stop nor plan the next round. The transient-retry path awaits
  asyncio.sleep(0) so a hot in-memory mock spin still yields to the event
  loop (cancel/supervisor/timeout stay effective).

Seams (monkeypatch targets for tests):
- from app.analysis import board as board_module, workers as workers_module;
  call workers_module.analyze_board / critique_board / assess_gaps and
  board_module.format_board via those module attributes.
- from app.tools import dispatch as dispatch_module; call
  dispatch_module.run_tool_round so patching app.tools.dispatch.run_tool_round
  takes effect (same module object).
"""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from app.analysis import board as board_module
from app.analysis import workers as workers_module
from app.evidence.models import (
    Claim,
    ClaimStatus,
    Investigation,
    InvestigationStatus,
    StatusReason,
)
from app.investigations import TERMINAL, manager, new_claim_id
from app.metrics import LOOP_STOPS
from app.tools import dispatch as dispatch_module
from app.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _stop(reason: str, investigation_id: str) -> None:
    """Count one loop-decided ending and log it."""
    LOOP_STOPS.labels(reason=reason).inc()
    logger.info(
        {"message": "Investigation loop stopped", "investigation_id": investigation_id, "reason": reason}
    )


async def _guard(
    awaitable: Awaitable[T], *, investigation_id: str, worker: str
) -> T | None:
    """Await one worker call; WorkerError becomes None (logged)."""
    try:
        return await awaitable
    except workers_module.WorkerError as exc:
        logger.warning(
            {
                "message": "Investigation loop: worker failed",
                "investigation_id": investigation_id,
                "worker": worker,
                "error": str(exc),
            }
        )
        return None


async def _ensure_gathering(inv: Investigation) -> Investigation | None:
    """Move PLANNED into GATHERING; GATHERING passes through."""
    if inv.status == InvestigationStatus.GATHERING:
        return inv
    if inv.status != InvestigationStatus.PLANNED:
        logger.warning(
            {
                "message": "Investigation loop: cannot enter GATHERING",
                "investigation_id": inv.id,
                "status": inv.status.value,
            }
        )
        return None
    try:
        moved = await manager.transition(inv.id, InvestigationStatus.GATHERING)
    except ValueError as exc:
        logger.warning(
            {
                "message": "Investigation loop: cannot enter GATHERING",
                "investigation_id": inv.id,
                "error": str(exc),
            }
        )
        return None
    if moved is None:
        logger.warning(
            {"message": "Investigation loop: investigation vanished", "investigation_id": inv.id}
        )
        return None
    return moved


async def run_investigation_loop(investigation_id: str) -> None:
    """Run the adaptive loop for one investigation. Never raises."""
    try:
        await _run_loop(investigation_id)
    except Exception as exc:  # noqa: BLE001 - background runner must never raise
        logger.warning(
            {
                "message": "Investigation loop failed",
                "investigation_id": investigation_id,
                "error": str(exc),
            }
        )


async def _run_loop(investigation_id: str) -> None:
    inv = await manager.get(investigation_id)
    if inv is None:
        logger.warning(
            {"message": "Investigation loop: investigation not found", "investigation_id": investigation_id}
        )
        return
    if inv.status in TERMINAL:
        return

    row = await manager.record_iteration(investigation_id)
    if row is None:
        logger.warning(
            {"message": "Investigation loop: investigation vanished", "investigation_id": investigation_id}
        )
        return
    if row.status in TERMINAL or row.status_reason is not None:
        # Round-zero iteration tripped a budget/wall-clock limit; the manager
        # already recorded the terminal state, so return without counting.
        return
    inv = row

    ensured = await _ensure_gathering(inv)
    if ensured is None:
        return
    inv = ensured

    baseline = [(name, inv.query) for name in dispatch_module.BASELINE_ROUND_ZERO]
    written, attempted, stopped = await dispatch_module.run_tool_round(investigation_id, baseline)
    if stopped:
        return
    if written == 0 and attempted:
        try:
            await manager.transition(
                investigation_id, InvestigationStatus.FAILED, StatusReason.PROVIDER_FAILURE
            )
        except ValueError as exc:
            logger.warning(
                {
                    "message": "Investigation loop: cannot mark PROVIDER_FAILURE",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )
            return
        _stop("provider_failure", investigation_id)
        return

    while True:
        cur = await manager.get(investigation_id)
        if cur is None:
            logger.warning(
                {
                    "message": "Investigation loop: investigation vanished",
                    "investigation_id": investigation_id,
                }
            )
            return
        if cur.status in TERMINAL:
            return
        if manager.cancel_event(investigation_id).is_set():
            return

        board_text = board_module.format_board(cur)
        query = cur.query

        analysis = await _guard(
            workers_module.analyze_board(board_text, query),
            investigation_id=investigation_id,
            worker="analysis",
        )
        if analysis is None:
            await asyncio.sleep(0)  # Yield so cancel/supervisor can fire on hot mock spins.
            continue
        critique = await _guard(
            workers_module.critique_board(board_text, query),
            investigation_id=investigation_id,
            worker="critique",
        )
        if critique is None:
            await asyncio.sleep(0)  # Same starvation guard as above.
            continue

        known_evidence_ids = {e.id for e in cur.board.evidence}
        known_claim_ids = {c.id for c in cur.board.claims}

        for draft in analysis.claims:
            filtered = [eid for eid in draft.evidence_ids if eid in known_evidence_ids]
            claim = Claim(
                id=new_claim_id(),
                investigation_id=investigation_id,
                statement=draft.statement,
                confidence=draft.confidence,
                evidence_ids=filtered,
            )
            try:
                added = await manager.add_claim(investigation_id, claim)
            except ValueError:
                return
            if added is None:
                return
            known_claim_ids.add(claim.id)

        for challenge in critique.challenges:
            target = challenge.target_claim_id
            if target is not None and target in known_claim_ids:
                try:
                    updated = await manager.set_claim_status(
                        investigation_id, target, ClaimStatus.CONTESTED
                    )
                except ValueError:
                    return
                if updated is None:
                    return
            else:
                standalone = Claim(
                    id=new_claim_id(),
                    investigation_id=investigation_id,
                    statement=challenge.point,
                    confidence=challenge.severity,
                    evidence_ids=[],
                    status=ClaimStatus.PROPOSED,
                )
                try:
                    added = await manager.add_claim(investigation_id, standalone)
                except ValueError:
                    return
                if added is None:
                    return
                known_claim_ids.add(standalone.id)

        gap = await _guard(
            workers_module.assess_gaps(board_text, query),
            investigation_id=investigation_id,
            worker="gap",
        )
        if gap is None:
            _stop("gap_error", investigation_id)
            return
        logger.info(
            {
                "message": "Investigation loop: gap assessment",
                "investigation_id": investigation_id,
                "sufficient": gap.sufficient,
                "rationale": gap.rationale,
            }
        )
        if gap.sufficient:
            _stop("sufficient", investigation_id)
            return

        grown = await manager.record_iteration(investigation_id)
        if grown is None:
            logger.warning(
                {
                    "message": "Investigation loop: investigation vanished",
                    "investigation_id": investigation_id,
                }
            )
            return
        if grown.status in TERMINAL or grown.status_reason is not None:
            return

        planned = [("radar_search", gap.radar_query), ("rag_retrieve", gap.rag_query)]
        written, attempted, stopped = await dispatch_module.run_tool_round(
            investigation_id, planned
        )
        if stopped:
            return
        if written == 0 and attempted:
            try:
                await manager.transition(
                    investigation_id, InvestigationStatus.FAILED, StatusReason.PROVIDER_FAILURE
                )
            except ValueError as exc:
                logger.warning(
                    {
                        "message": "Investigation loop: cannot mark PROVIDER_FAILURE",
                        "investigation_id": investigation_id,
                        "error": str(exc),
                    }
                )
                return
            _stop("provider_failure", investigation_id)
            return
        # Otherwise park-or-grow: loop around for the next analyze round.
