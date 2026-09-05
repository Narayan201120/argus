"""Stage P4-5b - investigation cost budget (DEC-053, backend only, mock-only).

All tests are mock-only: fakeredis persistence, scripted FakeTool results, and
no network. No live LLM fires (conftest auto-stubs the analysis workers
suite-wide; the loop test takes the scripted_milestone fixture).
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from prometheus_client import REGISTRY

from app.analysis import loop as loop_module
from app.analysis.loop import run_investigation_loop
from app.config import settings
from app.costs import estimate_llm_cost, web_fetch_cost, web_search_cost
from app.evidence.models import (
    SCHEMA_VERSION,
    BudgetLimits,
    BudgetUsage,
    Investigation,
    InvestigationStatus,
    StatusReason,
)
from app.evidence.store import EvidenceBoardStore
from app.investigations import InvestigationManager
from app.tools import dispatch as dispatch_module
from app.tools.base import BaseTool, EvidencePayload, ToolResult


@pytest.fixture
async def fake_redis() -> AsyncIterator[Any]:
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield fr
    await fr.aclose()


@pytest.fixture(autouse=True)
def _use_fake_redis(fake_redis: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", fake_redis)


class FakeTool(BaseTool):
    """Scripted stand-in; records every query it receives."""

    def __init__(self, name: str, items: list[EvidencePayload] | None = None) -> None:
        self.name: str = name
        self._items: list[EvidencePayload] = list(items) if items else []
        self.calls: int = 0
        self.queries: list[str] = []

    @property
    def enabled(self) -> bool:
        return True

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del params
        self.calls += 1
        self.queries.append(query)
        return ToolResult(tool_name=self.name, ok=True, items=list(self._items))


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> InvestigationManager:
    mgr = InvestigationManager(EvidenceBoardStore())
    monkeypatch.setattr(loop_module, "manager", mgr)
    monkeypatch.setattr(dispatch_module, "manager", mgr)
    return mgr


def _use_registry(monkeypatch: pytest.MonkeyPatch, registry: dict[str, BaseTool]) -> None:
    monkeypatch.setattr(dispatch_module, "build_tool_registry", lambda: registry)


def _direct_inv(*, cost_usd: float, deadline_at: float) -> Investigation:
    import time

    created = time.time()
    return Investigation(
        id="inv_direct",
        user_id="local",
        query="direct budget probe",
        status=InvestigationStatus.GATHERING,
        status_reason=None,
        created_at=created,
        updated_at=created,
        deadline_at=deadline_at,
        schema_version=SCHEMA_VERSION,
        budgets=BudgetLimits(max_iterations=5, max_tool_calls=10, max_wall_time_s=120),
        usage=BudgetUsage(cost_usd=cost_usd),
    )


# ── Price math ────────────────────────────────────────────────────────────────


def test_estimate_llm_cost_known_connector_matches_table() -> None:
    in_rate = settings.cost_usd_per_1k_input_tokens["openai"]
    out_rate = settings.cost_usd_per_1k_output_tokens["openai"]
    expected = 1000 / 1000.0 * in_rate + 500 / 1000.0 * out_rate
    assert estimate_llm_cost("openai", 1000, 500) == pytest.approx(expected)


def test_estimate_llm_cost_unknown_connector_uses_default() -> None:
    in_rate = settings.cost_usd_per_1k_input_tokens["default"]
    out_rate = settings.cost_usd_per_1k_output_tokens["default"]
    expected = 2000 / 1000.0 * in_rate + 1000 / 1000.0 * out_rate
    assert estimate_llm_cost("connector_nope_unknown", 2000, 1000) == pytest.approx(expected)


def test_estimate_llm_cost_zero_tokens_is_zero() -> None:
    assert estimate_llm_cost("openai", 0, 0) == 0.0
    assert estimate_llm_cost("connector_nope_unknown", 0, 0) == 0.0


def test_web_search_cost_matches_settings() -> None:
    assert web_search_cost() == pytest.approx(settings.cost_usd_per_web_search)


def test_web_fetch_cost_matches_settings() -> None:
    assert web_fetch_cost() == pytest.approx(settings.cost_usd_per_web_fetch)


# ── add_cost ──────────────────────────────────────────────────────────────────


async def test_add_cost_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    assert await mgr.add_cost("inv_nope_missing", 1.0) is None


async def test_add_cost_accumulates_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    inv = await mgr.create("accumulate probe", "local")
    try:
        first = await mgr.add_cost(inv.id, 0.05)
        assert first is not None
        assert first.usage.cost_usd == pytest.approx(0.05)
        second = await mgr.add_cost(inv.id, 0.07)
        assert second is not None
        assert second.usage.cost_usd == pytest.approx(0.12)
        assert second.status == InvestigationStatus.PLANNED
        assert second.status_reason is None
    finally:
        await mgr.cancel(inv.id)


async def test_add_cost_negative_clamps_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    inv = await mgr.create("negative probe", "local")
    try:
        row = await mgr.add_cost(inv.id, -5.0)
        assert row is not None
        assert row.usage.cost_usd == pytest.approx(0.0)
        assert row.status == InvestigationStatus.PLANNED
    finally:
        await mgr.cancel(inv.id)


async def test_add_cost_terminal_row_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    inv = await mgr.create("terminal probe", "local")
    try:
        assert await mgr.cancel(inv.id) is not None
        row = await mgr.add_cost(inv.id, 0.05)
        assert row is not None
        assert row.status == InvestigationStatus.CANCELLED
        assert row.usage.cost_usd == pytest.approx(0.0)
    finally:
        await mgr.cancel(inv.id)


async def test_add_cost_over_cap_trips_budget_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    cap = settings.investigation_max_cost_usd
    assert cap > 0
    inv = await mgr.create("over-cap probe", "local")
    try:
        row = await mgr.add_cost(inv.id, cap + 0.25)
        assert row is not None
        assert row.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert row.status_reason == StatusReason.COST_LIMIT
        # A trip stops further spend; it never refunds what accumulated.
        assert row.usage.cost_usd == pytest.approx(cap + 0.25)
    finally:
        await mgr.cancel(inv.id)


async def test_add_cost_cap_zero_disables_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "investigation_max_cost_usd", 0)
    mgr = _fresh_manager(monkeypatch)
    inv = await mgr.create("disabled-cap probe", "local")
    try:
        row = await mgr.add_cost(inv.id, 100.0)
        assert row is not None
        assert row.usage.cost_usd == pytest.approx(100.0)
        assert row.status == InvestigationStatus.PLANNED
        assert row.status_reason is None
    finally:
        await mgr.cancel(inv.id)


# ── check_budgets ─────────────────────────────────────────────────────────────


def test_check_budgets_over_cap_returns_cost_limit() -> None:
    import time

    cap = settings.investigation_max_cost_usd
    assert cap > 0
    inv = _direct_inv(cost_usd=cap + 1.0, deadline_at=time.time() + 120)
    assert InvestigationManager.check_budgets(inv) == StatusReason.COST_LIMIT


def test_check_budgets_wall_clock_still_first() -> None:
    import time

    cap = settings.investigation_max_cost_usd
    assert cap > 0
    inv = _direct_inv(cost_usd=cap + 1.0, deadline_at=time.time() - 1)
    assert InvestigationManager.check_budgets(inv) == StatusReason.WALL_CLOCK_LIMIT


# ── Old snapshots ─────────────────────────────────────────────────────────────


def test_budget_usage_legacy_snapshot_defaults_cost_zero() -> None:
    assert BudgetUsage.model_validate({}).cost_usd == 0.0


# ── Web reserve charges ───────────────────────────────────────────────────────


def _web_payload() -> EvidencePayload:
    return EvidencePayload(
        source_ref="web-ref-1", content="web finding", type="text", confidence=0.7
    )


async def test_web_reserve_charges_search_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_web_calls", 10)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)
    mgr = _fresh_manager(monkeypatch)
    web = FakeTool("web_search", items=[_web_payload()])
    _use_registry(monkeypatch, {"web_search": web})
    inv = await mgr.create("web reserve probe", "local")
    try:
        written, attempted, stopped = await dispatch_module.run_tool_round(
            inv.id, [("web_search", "reserve probe query")]
        )
        assert (written, attempted, stopped) == (1, True, False)
        assert web.calls == 1
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.usage.web_calls_used == 1
        assert loaded.usage.cost_usd == pytest.approx(web_search_cost())
    finally:
        await mgr.cancel(inv.id)


async def test_web_capped_accumulates_no_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_web_calls", 0)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)
    mgr = _fresh_manager(monkeypatch)
    web = FakeTool("web_search", items=[_web_payload()])
    _use_registry(monkeypatch, {"web_search": web})
    capped_labels = {"tool": "web_search", "status": "capped"}
    before = REGISTRY.get_sample_value("argus_tool_calls_total", capped_labels) or 0.0
    inv = await mgr.create("web capped probe", "local")
    try:
        written, attempted, stopped = await dispatch_module.run_tool_round(
            inv.id, [("web_search", "capped probe query")]
        )
        assert (written, attempted, stopped) == (0, False, False)
        assert web.calls == 0
        after = REGISTRY.get_sample_value("argus_tool_calls_total", capped_labels) or 0.0
        assert after == before + 1.0
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.usage.cost_usd == pytest.approx(0.0)
        assert loaded.usage.web_calls_used == 0
    finally:
        await mgr.cancel(inv.id)


# ── Loop-level: free investigations never trip ────────────────────────────────


async def test_free_loop_stays_zero_cost(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    del scripted_milestone
    monkeypatch.setattr(settings, "web_tools_enabled", False)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool(
                "radar_search",
                items=[
                    EvidencePayload(
                        source_ref="free-r0", content="free finding", type="text",
                        confidence=0.7,
                    )
                ],
            ),
            "rag_retrieve": FakeTool(
                "rag_retrieve",
                items=[
                    EvidencePayload(
                        source_ref="free-r1", content="free finding", type="text",
                        confidence=0.7,
                    )
                ],
            ),
        },
    )
    inv = await mgr.create("free loop probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.usage.cost_usd == pytest.approx(0.0)
    finally:
        await mgr.cancel(inv.id)
