"""Stage P4-3 - milestone synthesis test suite (mock-only, no network, no live LLM)."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import app.investigations as investigations_module
from app.analysis import events as events_module
from app.analysis import loop as loop_module
from app.analysis import synthesis as synthesis_module
from app.analysis import workers as workers_module
from app.analysis.loop import run_investigation_loop
from app.analysis.synthesis import SynthesisError, synthesis_store
from app.analysis.workers import AnalysisOutput, CritiqueOutput, GapOutput
from app.api.routes import investigations as investigations_route_module
from app.config import settings
from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.evidence.models import InvestigationStatus, StatusReason
from app.evidence.store import EvidenceBoardStore
from app.investigations import InvestigationManager
from app.main import app
from app.rediskit import holder
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
    monkeypatch.setattr(holder, "client", fake_redis)


class _FakeTool(BaseTool):
    """Scripted stand-in; records every query it receives."""

    def __init__(self, name: str, *, items: list[EvidencePayload] | None = None) -> None:
        self.name: str = name
        self._items: list[EvidencePayload] = list(items) if items else []
        self.queries: list[str] = []
        self.calls: int = 0

    @property
    def enabled(self) -> bool:
        return True

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del params
        self.calls += 1
        self.queries.append(query)
        return ToolResult(tool_name=self.name, ok=True, items=list(self._items))


class _FakeSynthesisConnector(BaseConnector):
    """Scripted connector exposing the real stream_query/query shape."""

    connector_id = "fake-synth"
    display_name = "Fake Synthesis Connector"
    capabilities: list[str] = ["text"]
    is_available = True

    def __init__(
        self,
        chunks: list[str] | None = None,
        *,
        query_content: str = "",
        stream_exc: Exception | None = None,
        query_exc: Exception | None = None,
    ) -> None:
        self._chunks: list[str] = list(chunks) if chunks else []
        self._query_content: str = query_content
        self._stream_exc: Exception | None = stream_exc
        self._query_exc: Exception | None = query_exc

    async def query(
        self, prompt: str, sub_query: str, config: ConnectorConfig
    ) -> ConnectorResponse:
        del prompt, config
        exc = self._query_exc
        if exc is not None:
            raise exc
        return ConnectorResponse(
            model_id="fake-model",
            content=self._query_content,
            latency_ms=5,
            token_usage=TokenUsage(1, 2, 3),
            status=ConnectorStatus.SUCCESS,
            error=None,
            sub_query=sub_query,
        )

    async def stream_query(
        self, prompt: str, sub_query: str, config: ConnectorConfig
    ) -> AsyncIterator[str]:
        del prompt, sub_query, config
        exc = self._stream_exc
        if exc is not None:
            raise exc
        for chunk in self._chunks:
            yield chunk

    async def health_check(self) -> bool:
        return True


def _payload(ref: str, content: str = "finding content") -> EvidencePayload:
    return EvidencePayload(source_ref=ref, content=content, type="text", confidence=0.7)


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> InvestigationManager:
    """Fresh manager patched into every binding direct tests can reach.

    test_analysis patches loop + dispatch only, which suffices for its
    scripted-milestone loops. These tests also drive the real run_milestone
    (reached via app.investigations.manager) and TestClient GET (reached via
    the investigations route's manager binding), so both are patched too.
    """
    mgr = InvestigationManager(EvidenceBoardStore())
    monkeypatch.setattr(loop_module, "manager", mgr)
    monkeypatch.setattr(dispatch_module, "manager", mgr)
    monkeypatch.setattr(investigations_module, "manager", mgr)
    monkeypatch.setattr(investigations_route_module, "manager", mgr)
    return mgr


def _use_registry(monkeypatch: pytest.MonkeyPatch, registry: dict[str, BaseTool]) -> None:
    monkeypatch.setattr(dispatch_module, "build_tool_registry", lambda: registry)


def _use_fake_connector(
    monkeypatch: pytest.MonkeyPatch, connector: _FakeSynthesisConnector
) -> None:
    monkeypatch.setattr(synthesis_module, "_pick_connector", lambda: connector)
    monkeypatch.setattr(synthesis_module, "_failover_candidate", lambda _exclude_id: None)


def _hist_count(sample: str) -> float:
    return REGISTRY.get_sample_value(sample) or 0.0


def _drain(queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return items


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = ""
        payload: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                loaded = json.loads(line[len("data:") :].strip())
                payload = loaded if isinstance(loaded, dict) else {"value": loaded}
        out.append((name, payload))
    return out


def _poll_complete(inv_id: str, deadline_s: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + deadline_s
    while True:
        got = client.get(f"/v1/investigate/{inv_id}")
        assert got.status_code == 200
        body: dict[str, Any] = got.json()
        if body["status"] == "complete":
            return body
        assert time.time() < deadline, "loop did not complete in time"
        time.sleep(0.05)


async def _empty_analyze(board_text: str, query: str) -> AnalysisOutput:
    del board_text, query
    return AnalysisOutput(claims=[])


async def _empty_critique(board_text: str, query: str) -> CritiqueOutput:
    del board_text, query
    return CritiqueOutput(challenges=[])


async def _sufficient_gap(board_text: str, query: str) -> GapOutput:
    del board_text, query
    return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="enough evidence")


async def _ok_board(
    board_text: str,
    query: str,
    milestone: int,
    emit: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    del board_text, query, milestone, emit
    return "# milestone report"


async def _boom_board(
    board_text: str,
    query: str,
    milestone: int,
    emit: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    del board_text, query, milestone, emit
    raise SynthesisError("provider_error: synthetic down")


def _script_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workers_module, "analyze_board", _empty_analyze)
    monkeypatch.setattr(workers_module, "critique_board", _empty_critique)
    monkeypatch.setattr(workers_module, "assess_gaps", _sufficient_gap)


# ── synthesize_board unit ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_board_streams_chunks_joined(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_connector(
        monkeypatch, _FakeSynthesisConnector(chunks=["# Title\n", "hello ", "world"])
    )
    result = await synthesis_module.synthesize_board("board text", "probe query", 0)
    assert result == "# Title\nhello world"


@pytest.mark.asyncio
async def test_synthesize_board_empty_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_connector(monkeypatch, _FakeSynthesisConnector(chunks=[], query_content=""))
    with pytest.raises(SynthesisError):
        await synthesis_module.synthesize_board("board text", "probe query", 0)


@pytest.mark.asyncio
async def test_synthesize_board_provider_exception_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_fake_connector(
        monkeypatch,
        _FakeSynthesisConnector(
            chunks=[],
            stream_exc=RuntimeError("provider boom"),
            query_exc=RuntimeError("provider boom"),
        ),
    )
    with pytest.raises(SynthesisError, match="provider_error"):
        await synthesis_module.synthesize_board("board text", "probe query", 0)


@pytest.mark.asyncio
async def test_synthesize_board_emit_receives_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fake_connector(monkeypatch, _FakeSynthesisConnector(chunks=["a", "b", "c"]))
    seen: list[str] = []

    async def _emit(delta: str) -> None:
        seen.append(delta)

    result = await synthesis_module.synthesize_board("board text", "probe query", 1, emit=_emit)
    assert result == "abc"
    assert seen == ["a", "b", "c"]


# ── run_milestone ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_milestone_stores_record_and_events(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _ok_board)
    inv = await mgr.create("synthesis probe query", "local")
    queue = events_module.subscribe(inv.id)
    try:
        result = await synthesis_module.run_milestone(inv.id, 2, False)
        assert result == "# milestone report"
        records = await synthesis_store.load(inv.id)
        assert len(records) == 1
        assert records[0].milestone == 2
        assert records[0].final is False
        assert records[0].markdown == "# milestone report"
        events = _drain(queue)
        kinds = [item["event"] for item in events]
        assert kinds[0] == events_module.SYNTHESIS_START
        assert kinds[-1] == events_module.SYNTHESIS_END
        assert events[0]["data"] == {"milestone": 2, "final": False}
        assert events[-1]["data"] == {"milestone": 2, "final": False}
    finally:
        events_module.unsubscribe(inv.id, queue)
        await mgr.cancel(inv.id)


@pytest.mark.asyncio
async def test_run_milestone_useful_answer_milestone_zero_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _ok_board)
    inv = await mgr.create("useful answer probe", "local")
    try:
        before = _hist_count("argus_time_to_useful_answer_seconds_count")
        assert await synthesis_module.run_milestone(inv.id, 0, False) == "# milestone report"
        assert _hist_count("argus_time_to_useful_answer_seconds_count") == before + 1.0
        assert await synthesis_module.run_milestone(inv.id, 1, False) == "# milestone report"
        assert _hist_count("argus_time_to_useful_answer_seconds_count") == before + 1.0
    finally:
        await mgr.cancel(inv.id)


@pytest.mark.asyncio
async def test_run_milestone_synthesis_error_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _boom_board)
    inv = await mgr.create("synthesis error probe", "local")
    queue = events_module.subscribe(inv.id)
    try:
        assert await synthesis_module.run_milestone(inv.id, 0, True) is None
        assert await synthesis_store.load(inv.id) == []
        kinds = [item["event"] for item in _drain(queue)]
        assert events_module.SYNTHESIS_START in kinds
        assert events_module.SYNTHESIS_END not in kinds
    finally:
        events_module.unsubscribe(inv.id, queue)
        await mgr.cancel(inv.id)


@pytest.mark.asyncio
async def test_run_milestone_missing_investigation_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_manager(monkeypatch)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _ok_board)
    assert await synthesis_module.run_milestone("inv_missing_nope", 0, False) is None


@pytest.mark.asyncio
async def test_run_milestone_terminal_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _ok_board)
    inv = await mgr.create("terminal probe", "local")
    queue = events_module.subscribe(inv.id)
    try:
        assert await mgr.cancel(inv.id) is not None
        assert await synthesis_module.run_milestone(inv.id, 0, True) is None
        assert await synthesis_store.load(inv.id) == []
        assert _drain(queue) == []
    finally:
        events_module.unsubscribe(inv.id, queue)
        await mgr.cancel(inv.id)


# ── Cancel mid-synthesis ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_mid_synthesis_returns_none_no_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    inv = await mgr.create("cancel synthesis probe", "local")

    async def _cancel_midway(
        board_text: str,
        query: str,
        milestone: int,
        emit: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        del board_text, query, milestone
        assert emit is not None
        await emit("part-one")
        mgr.cancel_event(inv.id).set()
        await emit("part-two")
        raise AssertionError("unreachable: cancel must abort streaming")

    monkeypatch.setattr(synthesis_module, "synthesize_board", _cancel_midway)
    queue = events_module.subscribe(inv.id)
    try:
        assert await synthesis_module.run_milestone(inv.id, 0, False) is None
        assert await synthesis_store.load(inv.id) == []
        kinds = [item["event"] for item in _drain(queue)]
        assert kinds[0] == events_module.SYNTHESIS_START
        assert events_module.SYNTHESIS_TOKEN in kinds
        assert events_module.SYNTHESIS_END not in kinds
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status not in {InvestigationStatus.CANCELLED}
    finally:
        events_module.unsubscribe(inv.id, queue)
        await mgr.cancel(inv.id)


# ── Completion mapping via loop ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_sufficient_completes_with_syntheses_on_get(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": _FakeTool("radar_search", items=[_payload("r0")]),
            "rag_retrieve": _FakeTool("rag_retrieve", items=[_payload("r1")]),
        },
    )
    _script_workers(monkeypatch)
    final_before = _hist_count("argus_time_to_final_report_seconds_count")
    inv = await mgr.create("completion probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.status_reason == StatusReason.SUFFICIENT_EVIDENCE
        assert scripted_milestone == [(0, True)]
        assert _hist_count("argus_time_to_final_report_seconds_count") == final_before + 1.0
        resp = client.get(f"/v1/investigate/{inv.id}")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["status"] == "complete"
        assert body["status_reason"] == "sufficient_evidence"
        assert len(body["syntheses"]) == 1
        assert body["syntheses"][0]["milestone"] == 0
        assert body["syntheses"][0]["final"] is True
        assert "scripted milestone 0" in body["syntheses"][0]["markdown"]
    finally:
        await mgr.cancel(inv.id)


# ── SSE ──────────────────────────────────────────────────────────────────────


def test_sse_snapshot_first_terminal_last(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    monkeypatch.setattr(settings, "radar_integration_enabled", False)
    monkeypatch.setattr(settings, "rag_integration_enabled", False)
    monkeypatch.setattr(settings, "web_tools_enabled", False)
    resp = client.post("/v1/investigate", json={"query": "sse probe query"})
    assert resp.status_code == 202
    inv_id: str = resp.json()["investigation_id"]
    try:
        body = _poll_complete(inv_id)
        assert len(body["syntheses"]) == 1
        assert scripted_milestone == [(0, True)]
        stream = client.get(f"/v1/investigate/{inv_id}/stream")
        assert stream.status_code == 200
        events = _parse_sse(stream.text)
        assert len(events) >= 2
        assert events[0][0] == events_module.BOARD_SNAPSHOT
        assert events[-1][0] == events_module.TERMINAL
        snapshot = events[0][1]
        assert snapshot["investigation_id"] == inv_id
        assert len(snapshot["syntheses"]) == 1
        assert "scripted milestone 0" in snapshot["syntheses"][0]["markdown"]
        assert events[-1][1] == {"status": "complete", "reason": "sufficient_evidence"}
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")


def test_sse_unknown_id_404() -> None:
    resp = client.get("/v1/investigate/inv_missing_nope/stream")
    assert resp.status_code == 404


# ── Fail-open ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_open_holder_client_none_stores_in_memory_and_get_shows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    monkeypatch.setattr(holder, "client", None)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _ok_board)
    inv = await mgr.create("fail-open probe", "local")
    try:
        assert await synthesis_module.run_milestone(inv.id, 0, False) == "# milestone report"
        records = await synthesis_store.load(inv.id)
        assert len(records) == 1
        resp = client.get(f"/v1/investigate/{inv.id}")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert len(body["syntheses"]) == 1
        assert body["syntheses"][0]["markdown"] == "# milestone report"
    finally:
        await mgr.cancel(inv.id)


# ── Synthesis failure still completes ────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesis_failure_still_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _fresh_manager(monkeypatch)
    _use_registry(
        monkeypatch,
        {
            "radar_search": _FakeTool("radar_search", items=[_payload("r0")]),
            "rag_retrieve": _FakeTool("rag_retrieve", items=[_payload("r1")]),
        },
    )
    _script_workers(monkeypatch)
    monkeypatch.setattr(synthesis_module, "synthesize_board", _boom_board)
    inv = await mgr.create("synthesis failure probe", "local")
    try:
        await run_investigation_loop(inv.id)
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert loaded.status_reason == StatusReason.SUFFICIENT_EVIDENCE
        assert await synthesis_store.load(inv.id) == []
    finally:
        await mgr.cancel(inv.id)
