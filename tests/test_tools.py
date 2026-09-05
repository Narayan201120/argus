"""Stage P4-1 - parallel tool dispatch opening round, mock-only."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.config import settings
from app.evidence.models import InvestigationStatus, StatusReason
from app.evidence.store import EvidenceBoardStore
from app.investigations import InvestigationManager
from app.main import app
from app.tools.base import BaseTool, EvidencePayload, ToolResult
from app.tools.dispatch import run_opening_round
from app.tools.radar import _map_paper, radar_search_tool
from app.tools.rag import rag_retrieve_tool
from app.tools.registry import build_tool_registry

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
    """Scripted stand-in for a baseline tool; records whether it ran."""

    def __init__(
        self,
        name: str,
        *,
        items: list[EvidencePayload] | None = None,
        ok: bool = True,
        delay_s: float = 0.0,
        enabled: bool = True,
        error: str | None = None,
    ) -> None:
        self.name: str = name
        self._items: list[EvidencePayload] = list(items) if items else []
        self._ok: bool = ok
        self._delay_s: float = delay_s
        self._enabled: bool = enabled
        self._error: str | None = error
        self.ran: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del query, params
        self.ran = True
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        return ToolResult(
            tool_name=self.name, ok=self._ok, items=list(self._items), error=self._error
        )


def _payload(
    ref: str, content: str = "finding content", kind: str = "text", confidence: float = 0.7
) -> EvidencePayload:
    return EvidencePayload(source_ref=ref, content=content, type=kind, confidence=confidence)


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> InvestigationManager:
    mgr = InvestigationManager(EvidenceBoardStore())
    monkeypatch.setattr("app.tools.dispatch.manager", mgr)
    return mgr


def _use_registry(monkeypatch: pytest.MonkeyPatch, registry: dict[str, BaseTool]) -> None:
    monkeypatch.setattr("app.tools.dispatch.build_tool_registry", lambda: registry)


# ── Contract ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_contract_writes_evidence_with_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool(
        "radar_search",
        items=[
            EvidencePayload(
                source_ref="radar://p1",
                content="paper one",
                type="radar_paper",
                confidence=0.9,
                provenance={"origin": "fake-radar"},
            )
        ],
    )
    rag = FakeTool(
        "rag_retrieve",
        items=[
            EvidencePayload(
                source_ref="rag://c1",
                content="chunk one",
                type="rag_chunk",
                confidence=0.6,
                provenance={"origin": "fake-rag"},
            )
        ],
    )
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag})
    inv = await mgr.create("contract probe query", "local")
    try:
        await run_opening_round(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.GATHERING
        assert len(loaded.board.evidence) == 2
        by_ref = {e.source_ref: e for e in loaded.board.evidence}
        paper = by_ref["radar://p1"]
        assert paper.investigation_id == inv.id
        assert paper.type == "radar_paper"
        assert paper.confidence == 0.9
        assert paper.provenance == {"origin": "fake-radar"}
        chunk = by_ref["rag://c1"]
        assert chunk.investigation_id == inv.id
        assert chunk.type == "rag_chunk"
        assert chunk.confidence == 0.6
        assert chunk.provenance == {"origin": "fake-rag"}
    finally:
        await mgr.cancel(inv.id)


# ── Concurrency ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_tools_run_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("ref-a")], delay_s=0.3),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("ref-b")], delay_s=0.3),
        },
    )
    inv = await mgr.create("concurrency probe query", "local")
    try:
        start = time.perf_counter()
        await run_opening_round(inv.id)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.55  # sequential would take >= 0.6s
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert {e.source_ref for e in loaded.board.evidence} == {"ref-a", "ref-b"}
    finally:
        await mgr.cancel(inv.id)


# ── Budget cutoff ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_cutoff_stops_second_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "investigation_max_tool_calls", 1)
    mgr = _fresh_manager(monkeypatch)
    first = FakeTool("radar_search", items=[_payload("ref-1")])
    second = FakeTool("rag_retrieve", items=[_payload("ref-2")])
    _use_registry(monkeypatch, {"radar_search": first, "rag_retrieve": second})
    inv = await mgr.create("budget probe query", "local")
    try:
        await run_opening_round(inv.id)
        assert second.ran is False
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert loaded.status_reason == StatusReason.TOOL_CALL_LIMIT
    finally:
        await mgr.cancel(inv.id)


# ── Cancel propagation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_propagates_and_skips_slow_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool(
                "radar_search", items=[_payload("slow-ref")], delay_s=5.0
            ),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("fast-ref")]),
        },
    )
    inv = await mgr.create("cancel probe query", "local")
    task: asyncio.Task[None] = asyncio.create_task(run_opening_round(inv.id))
    try:
        await asyncio.sleep(0.3)
        cancelled = await mgr.cancel(inv.id)
        assert cancelled is not None
        await asyncio.wait_for(task, timeout=10.0)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.CANCELLED
        assert all(e.source_ref != "slow-ref" for e in loaded.board.evidence)
    finally:
        if not task.done():
            task.cancel()
        await mgr.cancel(inv.id)


# ── Timeout ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_timeout_parks_with_healthy_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "tool_timeout_s", 1)
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool(
                "radar_search", items=[_payload("slow-ref")], delay_s=5.0
            ),
            "rag_retrieve": FakeTool("rag_retrieve", items=[_payload("healthy-ref")]),
        },
    )
    labels = {"tool": "radar_search", "status": "timeout"}
    before = REGISTRY.get_sample_value("argus_tool_calls_total", labels) or 0.0
    inv = await mgr.create("timeout probe query", "local")
    try:
        await run_opening_round(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.GATHERING
        assert {e.source_ref for e in loaded.board.evidence} == {"healthy-ref"}
        after = REGISTRY.get_sample_value("argus_tool_calls_total", labels) or 0.0
        assert after == before + 1.0
    finally:
        await mgr.cancel(inv.id)


# ── All-fail ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_fail_marks_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", ok=False, error="boom"),
            "rag_retrieve": FakeTool("rag_retrieve", ok=False, error="bust"),
        },
    )
    inv = await mgr.create("all-fail probe query", "local")
    try:
        await run_opening_round(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.FAILED
        assert loaded.status_reason == StatusReason.PROVIDER_FAILURE
    finally:
        await mgr.cancel(inv.id)


# ── Dedup ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_collapses_shared_source_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("shared-ref")]),
            "rag_retrieve": FakeTool(
                "rag_retrieve", items=[_payload("shared-ref", content="other words")]
            ),
        },
    )
    inv = await mgr.create("dedup probe query", "local")
    try:
        await run_opening_round(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert [e.source_ref for e in loaded.board.evidence] == ["shared-ref"]
    finally:
        await mgr.cancel(inv.id)


# ── Disabled-everything ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_everything_parks_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", enabled=False),
            "rag_retrieve": FakeTool("rag_retrieve", enabled=False),
        },
    )
    inv = await mgr.create("disabled probe query", "local")
    try:
        await run_opening_round(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.GATHERING
        assert loaded.board.evidence == []
        assert loaded.usage.tool_calls_used == 0
    finally:
        await mgr.cancel(inv.id)


# ── Registry ────────────────────────────────────────────────────────────────


def test_registry_returns_five_unique_tools_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "radar_integration_enabled", False)
    monkeypatch.setattr(settings, "rag_integration_enabled", False)
    monkeypatch.setattr(settings, "web_tools_enabled", False)
    registry = build_tool_registry()
    assert set(registry) == {
        "radar_search",
        "radar_similar",
        "rag_retrieve",
        "web_search",
        "web_fetch",
    }
    assert all(not tool.enabled for tool in registry.values())


def test_registry_rejects_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.tools import registry as registry_module

    monkeypatch.setattr(registry_module, "ALL_TOOLS", [FakeTool("dup"), FakeTool("dup")])
    with pytest.raises(ValueError, match="duplicate tool name"):
        build_tool_registry()


# ── Radar / RAG mapping and failure surface ─────────────────────────────────


def test_map_paper_shapes_evidence_payload() -> None:
    item = _map_paper(
        {"title": "Paper title", "abstract": "Abstract text", "url": "https://x", "score": 0.9},
        0,
        "radar_search",
        5,
    )
    assert item is not None
    assert item.source_ref == "https://x"
    assert item.type == "radar_paper"
    assert item.confidence == 0.9
    assert item.content == "Paper title\nAbstract text"
    assert _map_paper({"title": "", "abstract": ""}, 0, "radar_search", 1) is None
    assert _map_paper("not-a-dict", 0, "radar_search", 1) is None
    clamped = _map_paper({"title": "T", "score": 5.0}, 0, "radar_search", 1)
    assert clamped is not None
    assert clamped.confidence == 1.0


@pytest.mark.asyncio
async def test_radar_unreachable_returns_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "radar_integration_enabled", True)
    monkeypatch.setattr(settings, "radar_base_url", "http://127.0.0.1:9")
    monkeypatch.setattr(settings, "tool_timeout_s", 2)
    result = await radar_search_tool.run("unreachable probe")
    assert result.ok is False
    assert result.items == []


def test_rag_disabled_without_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_integration_enabled", True)
    monkeypatch.setattr(settings, "rag_service_user", None)
    monkeypatch.setattr(settings, "rag_service_pass", None)
    assert rag_retrieve_tool.enabled is False


# ── Route integration ───────────────────────────────────────────────────────


def test_route_runs_loop_and_completes(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    monkeypatch.setattr(settings, "radar_integration_enabled", False)
    monkeypatch.setattr(settings, "rag_integration_enabled", False)
    monkeypatch.setattr(settings, "web_tools_enabled", False)
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
            if body["status"] == "complete" or time.time() >= deadline:
                break
            time.sleep(0.05)
        assert body["status"] == "complete"
        assert body["status_reason"] == "sufficient_evidence"
        assert body["counts"] == {"evidence": 0, "claims": 0}
        assert len(body["syntheses"]) == 1
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")


# ── Fail-open ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_open_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", None)
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": FakeTool("radar_search", items=[_payload("mem-ref")]),
            "rag_retrieve": FakeTool("rag_retrieve", enabled=False),
        },
    )
    inv = await mgr.create("fail-open probe query", "local")
    try:
        await run_opening_round(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.GATHERING
        assert [e.source_ref for e in loaded.board.evidence] == ["mem-ref"]
    finally:
        await mgr.cancel(inv.id)
