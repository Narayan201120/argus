"""P4-0 investigation lifecycle manager (DEC-053, backend only).

Owns Investigation state transitions, budget accounting, per-investigation
cancel events, and wall-clock supervision. No network, no LLM, no tools yet:
P4-1+ will drive add_evidence/add_claim/record_* from the agent loop.

Restart note: supervisors are in-memory only and are spawned on create, so
investigations reloaded from Redis after a process restart get no supervisor
in P4-0. Wall-clock expiry for such rows is still enforced lazily on the
next record_tool_call/record_iteration call via check_budgets, until a later
phase adds supervisor rehydration.
"""

import asyncio
import contextlib
import time
import uuid

from app.config import settings
from app.evidence.models import (
    SCHEMA_VERSION,
    Board,
    BudgetLimits,
    BudgetUsage,
    Claim,
    ClaimStatus,
    Evidence,
    Investigation,
    InvestigationStatus,
    StatusReason,
)
from app.evidence.store import EvidenceBoardStore
from app.metrics import INVESTIGATION_COST, INVESTIGATIONS_TOTAL
from app.utils.logger import get_logger

logger = get_logger(__name__)

TERMINAL: set[InvestigationStatus] = {
    InvestigationStatus.COMPLETE,
    InvestigationStatus.CANCELLED,
    InvestigationStatus.FAILED,
    InvestigationStatus.BUDGET_EXHAUSTED,
}

TRANSITIONS: dict[InvestigationStatus, set[InvestigationStatus]] = {
    InvestigationStatus.PLANNED: {
        InvestigationStatus.GATHERING,
        InvestigationStatus.CANCELLED,
        InvestigationStatus.FAILED,
    },
    InvestigationStatus.GATHERING: {
        InvestigationStatus.ANALYZING,
        InvestigationStatus.CANCELLED,
        # P4-3: milestone synthesis concludes gathering straight into COMPLETE;
        # the final synthesis runs off the gathered board, so no ANALYZING hop.
        InvestigationStatus.COMPLETE,
        InvestigationStatus.FAILED,
        InvestigationStatus.BUDGET_EXHAUSTED,
    },
    InvestigationStatus.ANALYZING: {
        InvestigationStatus.SYNTHESIZING,
        InvestigationStatus.GATHERING,
        InvestigationStatus.COMPLETE,
        InvestigationStatus.CANCELLED,
        InvestigationStatus.FAILED,
        InvestigationStatus.BUDGET_EXHAUSTED,
    },
    InvestigationStatus.SYNTHESIZING: {
        InvestigationStatus.COMPLETE,
        InvestigationStatus.GATHERING,
        InvestigationStatus.CANCELLED,
        InvestigationStatus.FAILED,
        InvestigationStatus.BUDGET_EXHAUSTED,
    },
}


def new_investigation_id() -> str:
    return "inv_" + uuid.uuid4().hex[:16]


def new_evidence_id() -> str:
    return "ev_" + uuid.uuid4().hex[:16]


def new_claim_id() -> str:
    return "cl_" + uuid.uuid4().hex[:16]


class InvestigationManager:
    """In-memory lifecycle owner backed by EvidenceBoardStore persistence."""

    def __init__(self, store: EvidenceBoardStore | None = None) -> None:
        self._store: EvidenceBoardStore = store if store is not None else EvidenceBoardStore()
        self._events: dict[str, asyncio.Event] = {}
        self._supervisors: dict[str, asyncio.Task[None]] = {}

    async def create(self, query: str, user_id: str = "local") -> Investigation:
        query = query.strip()
        user_id = user_id.strip()
        if not query:
            raise ValueError("query must be non-empty")
        if not user_id:
            raise ValueError("user_id must be non-empty")
        now = time.time()
        inv = Investigation(
            id=new_investigation_id(),
            user_id=user_id,
            query=query,
            status=InvestigationStatus.PLANNED,
            status_reason=None,
            created_at=now,
            updated_at=now,
            deadline_at=now + max(settings.investigation_max_wall_time_s, 1),
            schema_version=SCHEMA_VERSION,
            budgets=BudgetLimits(
                max_iterations=settings.investigation_max_iterations,
                max_tool_calls=settings.investigation_max_tool_calls,
                max_wall_time_s=settings.investigation_max_wall_time_s,
            ),
            usage=BudgetUsage(iterations_used=0, tool_calls_used=0),
            board=Board(),
        )
        await self._store.save(inv, settings.investigation_ttl_s)
        self._events[inv.id] = asyncio.Event()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                {"message": "No running loop; skipping supervisor", "investigation_id": inv.id}
            )
        else:
            task = loop.create_task(self._supervise(inv.id))
            self._supervisors[inv.id] = task

            def _done(_task: asyncio.Task[None], key: str = inv.id) -> None:
                self._supervisors.pop(key, None)

            task.add_done_callback(_done)
        INVESTIGATIONS_TOTAL.labels(event="created").inc()
        logger.info({"message": "Investigation created", "investigation_id": inv.id})
        return inv

    async def get(self, investigation_id: str) -> Investigation | None:
        return await self._store.load(investigation_id)

    async def list_recent(self, limit: int = 20) -> list[Investigation]:
        clamped = max(1, min(int(limit), 100))
        return await self._store.list_recent(clamped)

    @staticmethod
    def check_budgets(inv: Investigation) -> StatusReason | None:
        if time.time() >= inv.deadline_at:
            return StatusReason.WALL_CLOCK_LIMIT
        if inv.usage.tool_calls_used > inv.budgets.max_tool_calls:
            return StatusReason.TOOL_CALL_LIMIT
        if inv.usage.iterations_used > inv.budgets.max_iterations:
            return StatusReason.ITERATION_LIMIT
        # Cost check runs last. Cap <= 0 disables it.
        cap = settings.investigation_max_cost_usd
        if cap > 0 and inv.usage.cost_usd > cap:
            return StatusReason.COST_LIMIT
        return None

    async def transition(
        self,
        investigation_id: str,
        to: InvestigationStatus,
        reason: StatusReason | None = None,
    ) -> Investigation | None:
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        allowed = TRANSITIONS.get(inv.status)
        if allowed is None or to not in allowed:
            raise ValueError(f"Illegal transition {inv.status} -> {to} for {investigation_id}")
        inv.status = to
        inv.status_reason = reason
        inv.updated_at = time.time()
        await self._store.save(inv, settings.investigation_ttl_s)
        if to in TERMINAL:
            self._finish(investigation_id)
        return inv

    async def record_tool_call(self, investigation_id: str) -> Investigation | None:
        return await self._record_use(investigation_id, tool_call=True)

    async def record_web_call(self, investigation_id: str) -> Investigation | None:
        return await self._record_use(investigation_id, tool_call=True, web_call=True)

    async def record_iteration(self, investigation_id: str) -> Investigation | None:
        return await self._record_use(investigation_id, tool_call=False)

    async def add_cost(self, investigation_id: str, amount: float) -> Investigation | None:
        """Accumulate estimated USD spend; trip to BUDGET_EXHAUSTED on cap breach.

        None when the investigation is missing. Terminal rows come back
        unchanged without accumulating. Otherwise adds max(amount, 0.0),
        stamps updated_at, then applies the cost-trip check only: when
        settings.investigation_max_cost_usd > 0 and usage.cost_usd exceeds
        the cap, the row flips to BUDGET_EXHAUSTED / COST_LIMIT and the
        supervisor is finished. A trip stops further spend; it never refunds
        what was already accumulated. Cap <= 0 disables the check.
        """
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        if inv.status in TERMINAL:
            return inv
        inv.usage.cost_usd += max(float(amount), 0.0)
        inv.updated_at = time.time()
        INVESTIGATION_COST.observe(max(float(amount), 0.0))
        cap = settings.investigation_max_cost_usd
        if cap > 0 and inv.usage.cost_usd > cap:
            inv.status = InvestigationStatus.BUDGET_EXHAUSTED
            inv.status_reason = StatusReason.COST_LIMIT
            inv.updated_at = time.time()
            self._finish(investigation_id)
        await self._store.save(inv, settings.investigation_ttl_s)
        return inv

    async def _record_use(
        self, investigation_id: str, *, tool_call: bool, web_call: bool = False
    ) -> Investigation | None:
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        if inv.status in TERMINAL:
            return inv
        if tool_call:
            inv.usage.tool_calls_used += 1
            if web_call:
                inv.usage.web_calls_used += 1
        else:
            inv.usage.iterations_used += 1
        inv.updated_at = time.time()
        reason = self.check_budgets(inv)
        if reason is not None:
            inv.status = InvestigationStatus.BUDGET_EXHAUSTED
            inv.status_reason = reason
            inv.updated_at = time.time()
            self._finish(investigation_id)
        await self._store.save(inv, settings.investigation_ttl_s)
        return inv

    async def add_evidence(
        self, investigation_id: str, evidence: Evidence
    ) -> Investigation | None:
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        if inv.status in TERMINAL:
            raise ValueError(f"Cannot add evidence to terminal investigation {investigation_id}")
        inv.board.evidence.append(evidence)
        inv.updated_at = time.time()
        await self._store.save(inv, settings.investigation_ttl_s)
        return inv

    async def add_claim(self, investigation_id: str, claim: Claim) -> Investigation | None:
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        if inv.status in TERMINAL:
            raise ValueError(f"Cannot add claim to terminal investigation {investigation_id}")
        inv.board.claims.append(claim)
        inv.updated_at = time.time()
        await self._store.save(inv, settings.investigation_ttl_s)
        return inv

    async def set_claim_status(
        self, investigation_id: str, claim_id: str, status: ClaimStatus
    ) -> Investigation | None:
        """Set one claim's status. None if investigation missing.

        Raises ValueError when the investigation is terminal or when no
        claim carries claim_id.
        """
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        if inv.status in TERMINAL:
            raise ValueError(f"Cannot set claim status on terminal investigation {investigation_id}")
        for claim in inv.board.claims:
            if claim.id == claim_id:
                claim.status = status
                inv.updated_at = time.time()
                await self._store.save(inv, settings.investigation_ttl_s)
                return inv
        raise ValueError(f"Unknown claim {claim_id} for investigation {investigation_id}")

    async def cancel(self, investigation_id: str) -> Investigation | None:
        inv = await self._store.load(investigation_id)
        if inv is None:
            return None
        if inv.status in TERMINAL:
            return inv
        inv.status = InvestigationStatus.CANCELLED
        inv.status_reason = StatusReason.CANCELLED
        inv.updated_at = time.time()
        await self._store.save(inv, settings.investigation_ttl_s)
        self._finish(investigation_id)
        INVESTIGATIONS_TOTAL.labels(event="cancelled").inc()
        logger.info({"message": "Investigation cancelled", "investigation_id": inv.id})
        return inv

    def _finish(self, investigation_id: str) -> None:
        """Signal terminal state: wake the supervisor and drop its task (best effort)."""
        event = self._events.get(investigation_id)
        if event is not None:
            event.set()
        task = self._supervisors.get(investigation_id)
        if task is not None and not task.done():
            with contextlib.suppress(Exception):
                task.cancel()

    async def _supervise(self, investigation_id: str) -> None:
        """Expire the investigation when its wall-clock deadline passes."""
        try:
            inv = await self._store.load(investigation_id)
            if inv is None:
                return
            timeout = max(0.0, inv.deadline_at - time.time())
            event = self._events.get(investigation_id)
            if event is None:
                event = asyncio.Event()
                self._events[investigation_id] = event
            try:
                await asyncio.wait_for(event.wait(), timeout)
                return  # Cancel event set; cancel()/transition() already recorded the outcome.
            except TimeoutError:
                current = await self._store.load(investigation_id)
                if current is None or current.status in TERMINAL:
                    return
                current.status = InvestigationStatus.BUDGET_EXHAUSTED
                current.status_reason = StatusReason.WALL_CLOCK_LIMIT
                current.updated_at = time.time()
                await self._store.save(current, settings.investigation_ttl_s)
                INVESTIGATIONS_TOTAL.labels(event="expired").inc()
                logger.info(
                    {"message": "Investigation expired", "investigation_id": investigation_id}
                )
        except asyncio.CancelledError:
            pass

    def cancel_event(self, investigation_id: str) -> asyncio.Event:
        return self._events.setdefault(investigation_id, asyncio.Event())

    async def sweep_expired(self) -> dict[str, int]:
        """Rehydrate supervisors after a restart; expire overdue rows. Never raises."""
        expired = 0
        rehydrated = 0
        try:
            rows = await self._store.list_recent(500)
        except Exception as exc:  # noqa: BLE001 - sweep never raises
            logger.warning(
                {"message": "Investigation sweep list failed (ignored)", "error": str(exc)}
            )
            return {"expired": 0, "rehydrated": 0}
        for row in rows:
            try:
                if row.status in TERMINAL:
                    continue
                task = self._supervisors.get(row.id)
                if task is not None and not task.done():
                    continue
                if row.id in self._events:
                    continue
                current = await self._store.load(row.id)
                if current is None or current.status in TERMINAL:
                    continue
                live = self._supervisors.get(current.id)
                if live is not None and not live.done():
                    continue
                if current.id in self._events:
                    continue
                if time.time() >= current.deadline_at:
                    current.status = InvestigationStatus.BUDGET_EXHAUSTED
                    current.status_reason = StatusReason.WALL_CLOCK_LIMIT
                    current.updated_at = time.time()
                    await self._store.save(current, settings.investigation_ttl_s)
                    expired += 1
                else:
                    self._events[current.id] = asyncio.Event()
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning(
                            {
                                "message": "No running loop; skipping supervisor",
                                "investigation_id": current.id,
                            }
                        )
                        self._events.pop(current.id, None)
                        continue
                    sup = loop.create_task(self._supervise(current.id))
                    self._supervisors[current.id] = sup

                    def _done(_task: asyncio.Task[None], key: str = current.id) -> None:
                        self._supervisors.pop(key, None)

                    sup.add_done_callback(_done)
                    rehydrated += 1
            except Exception as exc:  # noqa: BLE001 - per-row isolation, never raises
                logger.warning(
                    {
                        "message": "Investigation sweep row failed (ignored)",
                        "error": str(exc),
                        "investigation_id": getattr(row, "id", "?"),
                    }
                )
                continue
        return {"expired": expired, "rehydrated": rehydrated}


manager = InvestigationManager()
