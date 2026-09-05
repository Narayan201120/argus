"""Stage P4-2 - adaptive investigation loop, mock-only."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.analysis import loop as loop_module
from app.analysis import workers as workers_module
from app.analysis.board import BOARD_MAX_CHARS, format_board
from app.analysis.loop import run_investigation_loop
from app.analysis.workers import (
    AnalysisOutput,
    Challenge,
    ClaimDraft,
    CritiqueOutput,
    GapOutput,
    WorkerError,
)
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
from app.investigations import InvestigationManager
from app.main import app
from app.tools import dispatch as dispatch_module
from app.tools.base import BaseTool, EvidencePayload, ToolResult

client = TestClient(app)


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

    def __init__(
        self,
        name: str,
        *,
        items: list[EvidencePayload] | None = None,
        later_items: list[EvidencePayload] | None = None,
        unique_per_call: bool = False,
        ok: bool = True,
        delay_s: float = 0.0,
        enabled: bool = True,
        error: str | None = None,
        slow_from_call: int = 0,
        slow_s: float = 0.0,
    ) -> None:
        self.name: str = name
        self._items: list[EvidencePayload] = list(items) if items else []
        self._later_items: list[EvidencePayload] | None = (
            list(later_items) if later_items is not None else None
        )
        self._unique_per_call: bool = unique_per_call
        self._ok: bool = ok
        self._delay_s: float = delay_s
        self._enabled: bool = enabled
        self._error: str | None = error
        self._slow_from_call: int = slow_from_call
        self._slow_s: float = slow_s
        self.queries: list[str] = []
        self.calls: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del params
        self.calls += 1
        self.queries.append(query)
        if self._slow_s > 0 and self.calls >= self._slow_from_call:
            await asyncio.sleep(self._slow_s)
        elif self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._unique_per_call:
            items = [
                EvidencePayload(
                    source_ref=f"{self.name}-ref-{self.calls}",
                    content="finding content",
                    type="text",
                    confidence=0.7,
                )
            ]
        elif self._later_items is not None and self.calls > 1:
            items = list(self._later_items)
        else:
            items = list(self._items)
        return ToolResult(tool_name=self.name, ok=self._ok, items=items, error=self._error)


def _payload(
    ref: str, content: str = "finding content", kind: str = "text", confidence: float = 0.7
) -> EvidencePayload:
    return EvidencePayload(source_ref=ref, content=content, type=kind, confidence=confidence)


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> InvestigationManager:
    mgr = InvestigationManager(EvidenceBoardStore())
    monkeypatch.setattr(loop_module, "manager", mgr)
    monkeypatch.setattr(dispatch_module, "manager", mgr)
    return mgr


def _use_registry(monkeypatch: pytest.MonkeyPatch, registry: dict[str, BaseTool]) -> None:
    monkeypatch.setattr(dispatch_module, "build_tool_registry", lambda: registry)


async def _empty_analyze(board_text: str, query: str) -> AnalysisOutput:
    del board_text, query
    return AnalysisOutput(claims=[])


async def _empty_critique(board_text: str, query: str) -> CritiqueOutput:
    del board_text, query
    return CritiqueOutput(challenges=[])


async def _sufficient_gap(board_text: str, query: str) -> GapOutput:
    del board_text, query
    return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="enough evidence")


def _board_inv(n_evidence: int, n_claims: int, content: str = "e") -> Investigation:
    now = time.time()
    evidence = [
        Evidence(
            id=f"ev_{i:03d}",
            investigation_id="inv_board",
            source_ref=f"ref-{i}",
            content=content,
            type="text",
            confidence=0.5,
            created_at=now,
        )
        for i in range(n_evidence)
    ]
    claims = [
        Claim(
            id=f"cl_{i:03d}",
            investigation_id="inv_board",
            statement=f"claim statement {i}",
            confidence=0.5,
            evidence_ids=[],
        )
        for i in range(n_claims)
    ]
    return Investigation(
        id="inv_board",
        user_id="local",
        query="board query",
        status=InvestigationStatus.GATHERING,
        status_reason=None,
        created_at=now,
        updated_at=now,
        deadline_at=now + 120,
        schema_version=SCHEMA_VERSION,
        budgets=BudgetLimits(max_iterations=3, max_tool_calls=12, max_wall_time_s=120),
        usage=BudgetUsage(),
        board=Board(evidence=evidence, claims=claims),
    )


# ── Renderer ────────────────────────────────────────────────────────────────


def test_board_evidence_and_claim_caps() -> None:
    text = format_board(_board_inv(25, 25))
    assert "Evidence: 25 (showing 20)" in text
    assert "Claims: 25 (showing 20)" in text
    assert text.count("source=") == 20
    assert text.count("evidence_ids=[") == 20
    assert len(text) <= BOARD_MAX_CHARS


def test_board_content_and_total_char_caps() -> None:
    marker = "MARKER-NOT-RENDERED"
    text = format_board(_board_inv(1, 0, content="A" * 1000 + marker + "B" * 1000))
    assert "A" * 1000 in text
    assert marker not in text
    huge = format_board(_board_inv(25, 25, content="C" * 3000))
    assert len(huge) == BOARD_MAX_CHARS


def test_board_deterministic() -> None:
    inv = _board_inv(3, 2, content="stable content")
    assert format_board(inv) == format_board(inv)


def test_board_empty_renders_prompt() -> None:
    text = format_board(_board_inv(0, 0))
    assert text != ""
    assert "no evidence yet" in text


# ── Gap-driven seeding ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gap_driven_seeding_routes_followup_queries(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool("radar_search", unique_per_call=True)
    rag = FakeTool("rag_retrieve", unique_per_call=True)
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag})
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)
    gap_calls: dict[str, int] = {"n": 0}

    async def fake_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        gap_calls["n"] += 1
        if gap_calls["n"] == 1:
            return GapOutput(
                sufficient=False,
                radar_query="gap-radar-q",
                rag_query="gap-rag-q",
                rationale="need more",
            )
        return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="done")

    monkeypatch.setattr(workers_module, "assess_gaps", fake_gap)
    inv = await mgr.create("original query text", "local")
    try:
        await run_investigation_loop(inv.id)
        assert radar.queries == ["original query text", "gap-radar-q"]
        assert rag.queries == ["original query text", "gap-rag-q"]
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.status_reason == StatusReason.SUFFICIENT_EVIDENCE
        assert scripted_milestone == [(0, False), (1, True)]
    finally:
        await mgr.cancel(inv.id)


# ── Sufficient after round 0 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sufficient_after_round_zero_completes(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("r0")]),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("r1")]),
        },
    )
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)
    monkeypatch.setattr(workers_module, "assess_gaps", _sufficient_gap)
    labels = {"reason": "sufficient"}
    before = REGISTRY.get_sample_value("argus_loop_stops_total", labels) or 0.0
    inv = await mgr.create("sufficient probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.status_reason == StatusReason.SUFFICIENT_EVIDENCE
        assert loaded.usage.iterations_used == 1
        assert scripted_milestone == [(0, True)]
        after = REGISTRY.get_sample_value("argus_loop_stops_total", labels) or 0.0
        assert after == before + 1.0
    finally:
        await mgr.cancel(inv.id)


# ── Iteration cap ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_iteration_cap_exhausts_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "investigation_max_iterations", 3)
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool("radar_search", unique_per_call=True)
    rag = FakeTool("rag_retrieve", unique_per_call=True)
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag})
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)

    async def hungry_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        return GapOutput(
            sufficient=False, radar_query="more-radar", rag_query="more-rag", rationale="never enough"
        )

    monkeypatch.setattr(workers_module, "assess_gaps", hungry_gap)
    inv = await mgr.create("cap probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert loaded.status_reason == StatusReason.ITERATION_LIMIT
        # The tripping record increments before the manager marks the budget,
        # so max 3 ends at 4 rather than 3.
        assert loaded.usage.iterations_used == 4
        assert radar.calls == 3
    finally:
        await mgr.cancel(inv.id)


# ── Worker failure tolerance ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_failure_recovers_and_runs_round_one(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool("radar_search", unique_per_call=True)
    rag = FakeTool("rag_retrieve", unique_per_call=True)
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag})
    analyze_calls: dict[str, int] = {"n": 0}

    async def flaky_analyze(board_text: str, query: str) -> AnalysisOutput:
        del board_text, query
        analyze_calls["n"] += 1
        if analyze_calls["n"] <= 2:
            raise WorkerError("provider_error: flaky analyze")
        return AnalysisOutput(claims=[])

    monkeypatch.setattr(workers_module, "analyze_board", flaky_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)
    gap_calls: dict[str, int] = {"n": 0}

    async def fake_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        gap_calls["n"] += 1
        if gap_calls["n"] == 1:
            return GapOutput(
                sufficient=False,
                radar_query="gap-radar-q",
                rag_query="gap-rag-q",
                rationale="need more",
            )
        return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="done")

    monkeypatch.setattr(workers_module, "assess_gaps", fake_gap)
    inv = await mgr.create("flaky analyze probe", "local")
    try:
        await run_investigation_loop(inv.id)
        assert analyze_calls["n"] >= 3
        assert radar.queries[1] == "gap-radar-q"
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.status_reason == StatusReason.SUFFICIENT_EVIDENCE
        assert len(loaded.board.evidence) >= 2
    finally:
        await mgr.cancel(inv.id)


@pytest.mark.asyncio
async def test_gap_failure_ends_loop_with_gap_error(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("r0")]),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("r1")]),
        },
    )
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)

    async def bad_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        raise WorkerError("provider_error: gap down")

    monkeypatch.setattr(workers_module, "assess_gaps", bad_gap)
    labels = {"reason": "gap_error"}
    before = REGISTRY.get_sample_value("argus_loop_stops_total", labels) or 0.0
    sufficient_labels = {"reason": "sufficient"}
    sufficient_before = REGISTRY.get_sample_value("argus_loop_stops_total", sufficient_labels) or 0.0
    inv = await mgr.create("gap error probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.status_reason == StatusReason.SUFFICIENT_EVIDENCE
        assert {e.source_ref for e in loaded.board.evidence} == {"r0", "r1"}
        # Partial report still concludes: gap_error counted, sufficient not.
        assert scripted_milestone == [(0, True)]
        after = REGISTRY.get_sample_value("argus_loop_stops_total", labels) or 0.0
        assert after == before + 1.0
        sufficient_after = REGISTRY.get_sample_value("argus_loop_stops_total", sufficient_labels) or 0.0
        assert sufficient_after == sufficient_before
    finally:
        await mgr.cancel(inv.id)


# ── Critique ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critique_known_claim_flipped_to_contested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("r0")]),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("r1")]),
        },
    )
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)

    async def challenging(board_text: str, query: str) -> CritiqueOutput:
        del board_text, query
        return CritiqueOutput(
            challenges=[
                Challenge(
                    target_claim_id="cl_known00000001", point="weak support", severity=0.9
                )
            ]
        )

    monkeypatch.setattr(workers_module, "critique_board", challenging)
    monkeypatch.setattr(workers_module, "assess_gaps", _sufficient_gap)
    inv = await mgr.create("critique probe", "local")
    seeded = Claim(
        id="cl_known00000001",
        investigation_id=inv.id,
        statement="seeded claim",
        confidence=0.6,
        evidence_ids=[],
    )
    assert await mgr.add_claim(inv.id, seeded) is not None
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        by_id = {c.id: c for c in loaded.board.claims}
        assert by_id["cl_known00000001"].status == ClaimStatus.CONTESTED
    finally:
        await mgr.cancel(inv.id)


@pytest.mark.asyncio
async def test_critique_unknown_target_creates_standalone_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("r0")]),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("r1")]),
        },
    )
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)

    async def novel_challenge(board_text: str, query: str) -> CritiqueOutput:
        del board_text, query
        return CritiqueOutput(
            challenges=[
                Challenge(
                    target_claim_id="cl_missing00000001",
                    point="novel concern xyz",
                    severity=0.4,
                )
            ]
        )

    monkeypatch.setattr(workers_module, "critique_board", novel_challenge)
    monkeypatch.setattr(workers_module, "assess_gaps", _sufficient_gap)
    inv = await mgr.create("standalone probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        matches = [c for c in loaded.board.claims if c.statement == "novel concern xyz"]
        assert len(matches) == 1
        assert matches[0].status == ClaimStatus.PROPOSED
        assert matches[0].evidence_ids == []
    finally:
        await mgr.cancel(inv.id)


@pytest.mark.asyncio
async def test_set_claim_status_unit_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    assert await mgr.set_claim_status("inv_nope_missing", "cl_x", ClaimStatus.CONTESTED) is None
    inv = await mgr.create("claim status probe", "local")
    try:
        with pytest.raises(ValueError):
            await mgr.set_claim_status(inv.id, "cl_unknown", ClaimStatus.SUPPORTED)
        seeded = Claim(
            id="cl_unit00000001",
            investigation_id=inv.id,
            statement="unit claim",
            confidence=0.5,
            evidence_ids=[],
        )
        assert await mgr.add_claim(inv.id, seeded) is not None
        updated = await mgr.set_claim_status(inv.id, seeded.id, ClaimStatus.SUPPORTED)
        assert updated is not None
        assert updated.board.claims[0].status == ClaimStatus.SUPPORTED
        assert await mgr.cancel(inv.id) is not None
        with pytest.raises(ValueError):
            await mgr.set_claim_status(inv.id, seeded.id, ClaimStatus.REJECTED)
    finally:
        await mgr.cancel(inv.id)


# ── Dedup across rounds ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_across_rounds_single_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool(
        "radar_search",
        items=[_payload("dup-ref")],
        later_items=[_payload("dup-ref", content="same source again"), _payload("fresh-ref")],
    )
    _use_registry(monkeypatch, {"radar_search": radar})
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)
    gap_calls: dict[str, int] = {"n": 0}

    async def fake_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        gap_calls["n"] += 1
        if gap_calls["n"] == 1:
            return GapOutput(
                sufficient=False,
                radar_query="follow-up",
                rag_query="follow-up",
                rationale="need more",
            )
        return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="done")

    monkeypatch.setattr(workers_module, "assess_gaps", fake_gap)
    inv = await mgr.create("dedup probe", "local")
    try:
        await run_investigation_loop(inv.id)
        assert radar.calls == 2
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        refs = [e.source_ref for e in loaded.board.evidence]
        assert refs.count("dup-ref") == 1
        assert set(refs) == {"dup-ref", "fresh-ref"}
    finally:
        await mgr.cancel(inv.id)


# ── Cancel mid-loop ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_mid_loop_no_second_followup(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool("radar_search", unique_per_call=True, slow_from_call=2, slow_s=5.0)
    rag = FakeTool("rag_retrieve", unique_per_call=True)
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag})
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)

    async def hungry_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        return GapOutput(
            sufficient=False, radar_query="gap-radar", rag_query="gap-rag", rationale="more"
        )

    monkeypatch.setattr(workers_module, "assess_gaps", hungry_gap)
    inv = await mgr.create("cancel probe", "local")
    task: asyncio.Task[None] = asyncio.create_task(run_investigation_loop(inv.id))
    try:
        await asyncio.sleep(0.3)
        cancelled = await mgr.cancel(inv.id)
        assert cancelled is not None
        await asyncio.wait_for(task, timeout=10.0)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.CANCELLED
        assert loaded.usage.iterations_used <= 2
        assert radar.calls <= 2
    finally:
        if not task.done():
            task.cancel()
        await mgr.cancel(inv.id)


# ── Route integration ───────────────────────────────────────────────────────


def test_route_investigate_completes_with_scripted_claim(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    monkeypatch.setattr(settings, "radar_integration_enabled", False)
    monkeypatch.setattr(settings, "rag_integration_enabled", False)
    monkeypatch.setattr(settings, "web_tools_enabled", False)

    async def scripted_analyze(board_text: str, query: str) -> AnalysisOutput:
        del board_text, query
        return AnalysisOutput(
            claims=[ClaimDraft(statement="route-scripted-claim", confidence=0.8, evidence_ids=[])]
        )

    monkeypatch.setattr(workers_module, "analyze_board", scripted_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)
    monkeypatch.setattr(workers_module, "assess_gaps", _sufficient_gap)
    resp = client.post("/v1/investigate", json={"query": "route probe query"})
    assert resp.status_code == 202
    inv_id: str = resp.json()["investigation_id"]
    try:
        body: dict[str, Any] = {}
        deadline = time.time() + 5.0
        while True:
            got = client.get(f"/v1/investigate/{inv_id}")
            assert got.status_code == 200
            body = got.json()
            statements = [c["statement"] for c in body["claims"]]
            if body["status"] == "complete" and "route-scripted-claim" in statements:
                break
            assert time.time() < deadline, "loop did not complete with scripted claim in time"
            time.sleep(0.05)
        assert body["status"] == "complete"
        assert body["status_reason"] == "sufficient_evidence"
        assert scripted_milestone == [(0, True)]
        assert len(body["syntheses"]) == 1
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")
