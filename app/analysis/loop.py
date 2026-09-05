"""P4-3 adaptive investigation loop (DEC-053, backend only, mock-only).

Drives one investigation from PLANNED through repeated gather/analyze/
critique/gap rounds, with milestone synthesis after each assessed worker
pass and a final synthesis that concludes gathering as COMPLETE.

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
- P4-3 synthesis: every assessed worker pass ends in a synthesis milestone.
  Insufficient gap stores a non-final milestone and the loop continues with
  the next milestone number. Sufficient gap, or a gap WorkerError, stores
  the final synthesis (a partial report that states its limits in prose)
  and transitions GATHERING -> COMPLETE with reason SUFFICIENT_EVIDENCE;
  the reason means concluded-by-design in both cases, so gap_error counts
  only its own LOOP_STOPS reason, never sufficient. A synthesis miss (None)
  never marks FAILED and never raises: non-final misses retry on the next
  pass without incrementing the milestone, final misses still conclude.
- TERMINAL broadcast: the loop publishes one TERMINAL event per run through
  the investigation bus and never a duplicate (local terminal_published
  flag). Provider-failure and completion paths publish the row they just
  transitioned to; every other silent return re-loads the row and publishes
  only when it is already terminal (budget trips, cancels, supervisor
  expiry owned elsewhere).

Seams (monkeypatch targets for tests):
- from app.analysis import board as board_module, workers as workers_module;
  call workers_module.analyze_board / critique_board / assess_gaps and
  board_module.format_board via those module attributes.
- from app.tools import dispatch as dispatch_module; call
  dispatch_module.run_tool_round so patching app.tools.dispatch.run_tool_round
  takes effect (same module object).
- from app.analysis import synthesis as synthesis_module,
  events as events_module; call synthesis_module.run_milestone and
  events_module.publish via those module attributes so tests can stub the
  P4-3 synthesis runner and observe the event bus.
"""

import asyncio
import inspect
import time
from collections.abc import Awaitable
from typing import Any, TypeVar

from app.analysis import board as board_module
from app.analysis import events as events_module
from app.analysis import synthesis as synthesis_module
from app.analysis import workers as workers_module
from app.evidence.models import (
    Claim,
    ClaimStatus,
    Investigation,
    InvestigationStatus,
    StatusReason,
)
from app.investigations import TERMINAL, manager, new_claim_id
from app.metrics import LOOP_STOPS, TIME_TO_FINAL_REPORT
from app.tools import dispatch as dispatch_module
from app.utils.logger import get_logger

logger = get_logger(__name__)

#: Cap on source_refs carried in one EVIDENCE_ADDED event.
EVIDENCE_ADDED_MAX_REFS = 50

T = TypeVar("T")


def _stop(reason: str, investigation_id: str) -> None:
    """Count one loop-decided ending and log it."""
    LOOP_STOPS.labels(reason=reason).inc()
    logger.info(
        {"message": "Investigation loop stopped", "investigation_id": investigation_id, "reason": reason}
    )


def _worker_kwargs(fn: object, investigation_id: str) -> dict[str, str]:
    """investigation_id kwarg only when the worker accepts it.

    Keeps 2-arg test fakes working: the real workers take a keyword-only
    investigation_id, older fakes take (board_text, query) only.
    """
    try:
        params = inspect.signature(fn).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    if "investigation_id" in params:
        return {"investigation_id": investigation_id}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"investigation_id": investigation_id}
    return {}


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


async def _emit(investigation_id: str, event: str, payload: dict[str, Any]) -> None:
    """Publish one bus event. Best effort: never raises."""
    try:
        await events_module.publish(investigation_id, event, payload)
    except Exception as exc:  # noqa: BLE001 - event broadcast must not break the loop
        logger.warning(
            {
                "message": "Investigation loop: event publish failed",
                "investigation_id": investigation_id,
                "event": event,
                "error": str(exc),
            }
        )


def _evidence_ids(inv: Investigation) -> set[str]:
    return {e.id for e in inv.board.evidence}


async def _emit_evidence_added(
    investigation_id: str, before: set[str], after: Investigation | None
) -> None:
    """Diff evidence ids across one tool round and broadcast new source_refs."""
    if after is None:
        return
    refs = [e.source_ref for e in after.board.evidence if e.id not in before]
    if not refs:
        return
    await _emit(
        investigation_id,
        events_module.EVIDENCE_ADDED,
        {"items": refs[:EVIDENCE_ADDED_MAX_REFS]},
    )


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

    async def _run_milestone(milestone: int, *, final: bool) -> str | None:
        """Call the synthesis runner; misses come back None and never raise."""
        try:
            return await synthesis_module.run_milestone(investigation_id, milestone, final)
        except Exception as exc:  # noqa: BLE001 - synthesis must never break the loop
            logger.warning(
                {
                    "message": "Investigation loop: synthesis call failed",
                    "investigation_id": investigation_id,
                    "milestone": milestone,
                    "final": final,
                    "error": str(exc),
                }
            )
            return None

    terminal_published = False

    async def _publish_terminal_if_row_terminal() -> bool:
        """Broadcast TERMINAL once when the stored row is already terminal."""
        nonlocal terminal_published
        if terminal_published:
            return True
        row = await manager.get(investigation_id)
        if row is None or row.status not in TERMINAL:
            return False
        reason = row.status_reason.value if row.status_reason is not None else row.status.value
        await _emit(
            investigation_id,
            events_module.TERMINAL,
            {"status": row.status.value, "reason": reason},
        )
        terminal_published = True
        return True

    async def _finish(milestone: int, *, final: bool) -> bool:
        """Store one synthesis milestone, then conclude gathering as COMPLETE.

        A synthesis miss (None) still concludes: the board stands on its own
        and a miss never marks FAILED. Returns True when synthesis stored a
        report. Emits nothing directly; the synthesis runner broadcasts its
        own start/token/end events, and the TERMINAL below goes out via the
        freshly stored row so a concurrent cancel/budget terminal wins over
        a stale "complete".
        """
        markdown = await _run_milestone(milestone, final=final)
        if markdown is None:
            logger.warning(
                {
                    "message": "Investigation loop: synthesis produced no report",
                    "investigation_id": investigation_id,
                    "milestone": milestone,
                    "final": final,
                }
            )
        try:
            await manager.transition(
                investigation_id, InvestigationStatus.COMPLETE, StatusReason.SUFFICIENT_EVIDENCE
            )
        except ValueError as exc:
            logger.warning(
                {
                    "message": "Investigation loop: cannot conclude gathering",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )
            await _publish_terminal_if_row_terminal()
            return False
        row = await manager.get(investigation_id)
        if row is not None:
            TIME_TO_FINAL_REPORT.observe(max(time.time() - row.created_at, 0.0))
        await _publish_terminal_if_row_terminal()
        return markdown is not None

    if inv.status in TERMINAL:
        await _publish_terminal_if_row_terminal()
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
        await _publish_terminal_if_row_terminal()
        return
    inv = row

    ensured = await _ensure_gathering(inv)
    if ensured is None:
        await _publish_terminal_if_row_terminal()
        return
    inv = ensured

    baseline = [(name, inv.query) for name in dispatch_module.BASELINE_ROUND_ZERO]
    pre_round_evidence = _evidence_ids(inv)
    await _emit(
        investigation_id,
        events_module.ROUND_STARTED,
        {"round": inv.usage.iterations_used, "queries": [planned_query for _, planned_query in baseline]},
    )
    written, attempted, stopped = await dispatch_module.run_tool_round(investigation_id, baseline)
    await _emit_evidence_added(
        investigation_id, pre_round_evidence, await manager.get(investigation_id)
    )
    if stopped:
        await _publish_terminal_if_row_terminal()
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
            await _publish_terminal_if_row_terminal()
            return
        _stop("provider_failure", investigation_id)
        await _publish_terminal_if_row_terminal()
        return

    milestone = 0
    while True:
        cur = await manager.get(investigation_id)
        if cur is None:
            logger.warning(
                {
                    "message": "Investigation loop: investigation vanished",
                    "investigation_id": investigation_id,
                }
            )
            await _publish_terminal_if_row_terminal()
            return
        if cur.status in TERMINAL:
            await _publish_terminal_if_row_terminal()
            return
        if manager.cancel_event(investigation_id).is_set():
            # cancel() flips the stored row terminal synchronously, so the
            # helper usually broadcasts cancelled/cancelled from the row;
            # when the row is not terminal yet there is nothing to broadcast.
            await _publish_terminal_if_row_terminal()
            return

        board_text = board_module.format_board(cur)
        query = cur.query

        analysis = await _guard(
            workers_module.analyze_board(
                board_text, query, **_worker_kwargs(workers_module.analyze_board, investigation_id)
            ),
            investigation_id=investigation_id,
            worker="analysis",
        )
        if analysis is None:
            await asyncio.sleep(0)  # Yield so cancel/supervisor can fire on hot mock spins.
            continue
        critique = await _guard(
            workers_module.critique_board(
                board_text, query, **_worker_kwargs(workers_module.critique_board, investigation_id)
            ),
            investigation_id=investigation_id,
            worker="critique",
        )
        if critique is None:
            await asyncio.sleep(0)  # Same starvation guard as above.
            continue

        known_evidence_ids = {e.id for e in cur.board.evidence}
        known_claim_ids = {c.id for c in cur.board.claims}
        pre_claim_ids = set(known_claim_ids)

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
                await _publish_terminal_if_row_terminal()
                return
            if added is None:
                await _publish_terminal_if_row_terminal()
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
                    await _publish_terminal_if_row_terminal()
                    return
                if updated is None:
                    await _publish_terminal_if_row_terminal()
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
                    await _publish_terminal_if_row_terminal()
                    return
                if added is None:
                    await _publish_terminal_if_row_terminal()
                    return
                known_claim_ids.add(standalone.id)

        await _emit(
            investigation_id,
            events_module.CLAIMS_UPDATED,
            {"count": len(known_claim_ids - pre_claim_ids)},
        )

        gap = await _guard(
            workers_module.assess_gaps(
                board_text, query, **_worker_kwargs(workers_module.assess_gaps, investigation_id)
            ),
            investigation_id=investigation_id,
            worker="gap",
        )
        if gap is None:
            # No gap verdict: store the final synthesis as a partial report
            # that states its limits in prose, then conclude by design under
            # the same SUFFICIENT_EVIDENCE reason. Counts gap_error only.
            await _finish(milestone, final=True)
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
            ok = await _finish(milestone, final=True)
            logger.info(
                {
                    "message": "Investigation loop: final synthesis stored",
                    "investigation_id": investigation_id,
                    "milestone": milestone,
                    "synthesis_ok": ok,
                }
            )
            _stop("sufficient", investigation_id)
            return

        milestone_markdown = await _run_milestone(milestone, final=False)
        if milestone_markdown is None:
            # Miss without increment: the next pass retries the same milestone.
            logger.warning(
                {
                    "message": "Investigation loop: milestone synthesis produced no report",
                    "investigation_id": investigation_id,
                    "milestone": milestone,
                }
            )
        else:
            milestone += 1

        grown = await manager.record_iteration(investigation_id)
        if grown is None:
            logger.warning(
                {
                    "message": "Investigation loop: investigation vanished",
                    "investigation_id": investigation_id,
                }
            )
            await _publish_terminal_if_row_terminal()
            return
        if grown.status in TERMINAL or grown.status_reason is not None:
            await _publish_terminal_if_row_terminal()
            return

        planned = [
            ("radar_search", gap.radar_query),
            ("rag_retrieve", gap.rag_query),
            ("web_search", gap.web_query),
        ]
        pre_round_evidence = _evidence_ids(grown)
        await _emit(
            investigation_id,
            events_module.ROUND_STARTED,
            {"round": grown.usage.iterations_used, "queries": [gap.radar_query, gap.rag_query]},
        )
        written, attempted, stopped = await dispatch_module.run_tool_round(
            investigation_id, planned
        )
        await _emit_evidence_added(
            investigation_id, pre_round_evidence, await manager.get(investigation_id)
        )
        if stopped:
            await _publish_terminal_if_row_terminal()
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
                await _publish_terminal_if_row_terminal()
                return
            _stop("provider_failure", investigation_id)
            await _publish_terminal_if_row_terminal()
            return
        # Otherwise grow: loop around for the next analyze round.
