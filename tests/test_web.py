"""Stage P4-5a - Tavily/fetch mapping, follow-up-only web calls, and web-call cap (mock-only).

All tests use fake httpx clients, scripted trafilatura extracts, and fakeredis.
No network, no live Tavily, no live LLM (conftest auto-stubs the analysis
workers suite-wide; loop tests take the scripted_milestone fixture).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx
import pytest
import trafilatura
from fakeredis import aioredis as fakeredis_aioredis
from prometheus_client import REGISTRY

from app.analysis import loop as loop_module
from app.analysis import workers as workers_module
from app.analysis.loop import run_investigation_loop
from app.analysis.workers import GapOutput
from app.config import settings
from app.evidence.models import InvestigationStatus
from app.evidence.store import EvidenceBoardStore
from app.investigations import InvestigationManager
from app.tools import dispatch as dispatch_module
from app.tools import web as web_module
from app.tools.base import BaseTool, EvidencePayload, ToolResult
from app.tools.web import (
    ANSWER_MAX_CHARS,
    MAX_FETCH_BYTES,
    RESULT_MAX_CHARS,
    TAVILY_SEARCH_URL,
    TavilySearchTool,
    WebFetchTool,
)


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
    """Scripted stand-in; fresh source_ref per call so every round writes."""

    def __init__(
        self,
        name: str,
        *,
        unique_per_call: bool = False,
        items: list[EvidencePayload] | None = None,
    ) -> None:
        self.name: str = name
        self._unique_per_call = unique_per_call
        self._items: list[EvidencePayload] = list(items) if items else []
        self.calls: int = 0
        self.ran: bool = False
        self.queries: list[str] = []

    @property
    def enabled(self) -> bool:
        return True

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del params
        self.calls += 1
        self.ran = True
        self.queries.append(query)
        if self._unique_per_call:
            items = [
                EvidencePayload(
                    source_ref=f"{self.name}-ref-{self.calls}",
                    content="finding content",
                    type="text",
                    confidence=0.7,
                )
            ]
        else:
            items = list(self._items)
        return ToolResult(tool_name=self.name, ok=True, items=items)


def _fresh_manager(monkeypatch: pytest.MonkeyPatch) -> InvestigationManager:
    mgr = InvestigationManager(EvidenceBoardStore())
    monkeypatch.setattr(loop_module, "manager", mgr)
    monkeypatch.setattr(dispatch_module, "manager", mgr)
    return mgr


def _use_registry(monkeypatch: pytest.MonkeyPatch, registry: dict[str, BaseTool]) -> None:
    def _build() -> dict[str, BaseTool]:
        return registry

    monkeypatch.setattr(dispatch_module, "build_tool_registry", _build)


# ── Tavily fakes ────────────────────────────────────────────────────────────


class _TavilyResponse:
    """Minimal stand-in for httpx.Response covering what TavilySearchTool reads."""

    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload


class _TavilyClient:
    """Fake httpx.AsyncClient recording Tavily POSTs."""

    def __init__(
        self, response: _TavilyResponse | None = None, exc: Exception | None = None
    ) -> None:
        self._response = response
        self._exc = exc
        self.post_urls: list[str] = []
        self.posts: list[Any] = []

    async def __aenter__(self) -> _TavilyClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    async def post(self, url: str, **kwargs: Any) -> _TavilyResponse:
        self.post_urls.append(url)
        self.posts.append(kwargs.get("json"))
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


def _install_tavily(
    monkeypatch: pytest.MonkeyPatch,
    response: _TavilyResponse | None = None,
    exc: Exception | None = None,
) -> _TavilyClient:
    client = _TavilyClient(response=response, exc=exc)

    def _create(*args: Any, **kwargs: Any) -> _TavilyClient:
        return client

    monkeypatch.setattr(web_module.httpx, "AsyncClient", _create)
    return client


def _tavily_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tavily_api_key", "test-key")
    monkeypatch.setattr(settings, "web_tools_enabled", True)
    monkeypatch.setattr(settings, "tool_timeout_s", 2)


async def test_tavily_answer_first_and_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    payload = {
        "answer": "  Synthesised answer text  ",
        "results": [
            {"url": "https://example.com/a", "title": "A title", "content": "A snippet", "score": 0.9},
            {"url": "https://example.com/b", "title": "B title", "content": "B snippet"},
        ],
    }
    client = _install_tavily(monkeypatch, _TavilyResponse(200, payload))
    result = await TavilySearchTool().run("probe query")
    assert result.ok is True
    assert result.tool_name == "web_search"
    assert [item.type for item in result.items] == ["web_answer", "web_result", "web_result"]
    assert result.items[0].source_ref == "tavily:answer"
    assert result.items[0].content == "Synthesised answer text"
    assert result.items[0].confidence == 0.7
    assert result.items[1].source_ref == "https://example.com/a"
    assert result.items[1].confidence == 0.9
    assert result.items[2].confidence == pytest.approx(0.5)
    assert client.post_urls == [TAVILY_SEARCH_URL]
    body = client.posts[0]
    assert body["api_key"] == "test-key"
    assert body["query"] == "probe query"
    assert body["max_results"] == 5
    assert body["include_answer"] is True


async def test_tavily_score_clamp_and_rank_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    payload = {
        "results": [
            {"url": "https://example.com/high", "title": "H", "content": "C", "score": 9.9},
            {"url": "https://example.com/plain", "title": "P", "content": "C"},
            {"url": "https://example.com/neg", "title": "N", "content": "C", "score": -2.0},
        ]
    }
    _install_tavily(monkeypatch, _TavilyResponse(200, payload))
    result = await TavilySearchTool().run("clamp probe")
    assert result.ok is True
    assert [item.confidence for item in result.items] == [1.0, 0.5, 0.0]


async def test_tavily_skips_url_less_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    payload = {
        "results": [
            {"title": "no url", "content": "dropped"},
            "not-a-dict",
            {"url": "https://example.com/empty", "title": "", "content": "   "},
            {"url": "https://example.com/kept", "title": "K", "content": "kept snippet"},
        ]
    }
    _install_tavily(monkeypatch, _TavilyResponse(200, payload))
    result = await TavilySearchTool().run("skip probe")
    assert result.ok is True
    assert [item.source_ref for item in result.items] == ["https://example.com/kept"]


async def test_tavily_truncation_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    payload = {
        "answer": "A" * 5000,
        "results": [{"url": "https://example.com/long", "title": "T", "content": "C" * 5000}],
    }
    _install_tavily(monkeypatch, _TavilyResponse(200, payload))
    result = await TavilySearchTool().run("long probe")
    assert result.ok is True
    assert len(result.items) == 2
    assert len(result.items[0].content) <= ANSWER_MAX_CHARS
    assert len(result.items[1].content) <= RESULT_MAX_CHARS


async def test_tavily_bare_list_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    payload = [{"url": "https://example.com/only", "title": "Only", "content": "Body"}]
    _install_tavily(monkeypatch, _TavilyResponse(200, payload))
    result = await TavilySearchTool().run("bare probe")
    assert result.ok is True
    assert len(result.items) == 1
    assert result.items[0].type == "web_result"
    assert result.items[0].source_ref == "https://example.com/only"


@pytest.mark.parametrize("status", [429, 500])
async def test_tavily_http_error_surface(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    _install_tavily(monkeypatch, _TavilyResponse(status, None, text="busy"))
    result = await TavilySearchTool().run("error probe")
    assert result.ok is False
    assert result.items == []
    assert str(status) in (result.error or "")


async def test_tavily_timeout_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    _tavily_env(monkeypatch)
    _install_tavily(monkeypatch, None, httpx.ConnectTimeout("slow link"))
    result = await TavilySearchTool().run("timeout probe")
    assert result.ok is False
    assert result.items == []
    assert "tavily request failed" in (result.error or "")


async def test_tavily_missing_key_no_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "web_tools_enabled", True)
    created: list[bool] = []

    def _create(*args: Any, **kwargs: Any) -> Any:
        created.append(True)
        raise AssertionError("missing key must not touch HTTP")

    monkeypatch.setattr(web_module.httpx, "AsyncClient", _create)
    result = await TavilySearchTool().run("anything")
    assert result.ok is False
    assert "not configured" in (result.error or "")
    assert created == []


# ── Fetch fakes ─────────────────────────────────────────────────────────────


class _FetchStream:
    """Fake httpx streaming response covering what WebFetchTool reads."""

    def __init__(
        self,
        body: bytes,
        status: int = 200,
        content_length: str | None = None,
        charset: str = "utf-8",
    ) -> None:
        self._body = body
        self.status_code = status
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["content-length"] = content_length
        self.charset_encoding = charset

    async def __aenter__(self) -> _FetchStream:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    async def aiter_bytes(self, chunk_size: int = 65536) -> AsyncGenerator[bytes, None]:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


class _FetchClient:
    """Fake httpx.AsyncClient recording fetch stream calls."""

    def __init__(
        self, stream: _FetchStream | None = None, exc: Exception | None = None
    ) -> None:
        self._stream = stream
        self._exc = exc
        self.stream_calls: list[tuple[str, str]] = []

    async def __aenter__(self) -> _FetchClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def stream(self, method: str, url: str) -> _FetchStream:
        self.stream_calls.append((method, url))
        if self._exc is not None:
            raise self._exc
        assert self._stream is not None
        return self._stream


def _install_fetch(
    monkeypatch: pytest.MonkeyPatch,
    stream: _FetchStream | None = None,
    exc: Exception | None = None,
) -> _FetchClient:
    client = _FetchClient(stream=stream, exc=exc)

    def _create(*args: Any, **kwargs: Any) -> _FetchClient:
        return client

    monkeypatch.setattr(web_module.httpx, "AsyncClient", _create)
    monkeypatch.setattr(settings, "tool_timeout_s", 2)
    return client


def _script_extract(monkeypatch: pytest.MonkeyPatch, text: str | None) -> list[str]:
    calls: list[str] = []

    def _fake(html: str, *args: Any, **kwargs: Any) -> str | None:
        calls.append(html)
        return text

    monkeypatch.setattr(trafilatura, "extract", _fake)
    return calls


async def test_fetch_maps_page(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _FetchStream(b"<html><body><p>Hello fetch world</p></body></html>", status=200)
    client = _install_fetch(monkeypatch, stream)
    extract_calls = _script_extract(monkeypatch, "Readable page text")
    result = await WebFetchTool().run("need page", params={"url": "https://example.com/article"})
    assert result.ok is True
    assert len(result.items) == 1
    item = result.items[0]
    assert item.type == "web_page"
    assert item.source_ref == "https://example.com/article"
    assert item.content == "Readable page text"
    assert item.confidence == 0.6
    assert client.stream_calls == [("GET", "https://example.com/article")]
    assert len(extract_calls) == 1


async def test_fetch_query_as_url_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _FetchStream(b"<html><body>hi</body></html>", status=200)
    client = _install_fetch(monkeypatch, stream)
    _script_extract(monkeypatch, "Query url text")
    result = await WebFetchTool().run("https://example.com/from-query")
    assert result.ok is True
    assert result.items[0].source_ref == "https://example.com/from-query"
    assert client.stream_calls == [("GET", "https://example.com/from-query")]


async def test_fetch_url_required_without_http(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[bool] = []

    def _create(*args: Any, **kwargs: Any) -> Any:
        created.append(True)
        raise AssertionError("missing url must not touch HTTP")

    monkeypatch.setattr(web_module.httpx, "AsyncClient", _create)
    result = await WebFetchTool().run("bare topic")
    assert result.ok is False
    assert "url required" in (result.error or "")
    assert created == []


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://localhost:8000/page",
        "http://127.0.0.1:9/",
        "http://10.1.2.3/internal",
        "ftp://example.com/file",
    ],
)
async def test_fetch_ssrf_refused_without_http(
    bad_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[bool] = []

    def _create(*args: Any, **kwargs: Any) -> Any:
        created.append(True)
        raise AssertionError("refused url must not touch HTTP")

    monkeypatch.setattr(web_module.httpx, "AsyncClient", _create)
    result = await WebFetchTool().run("need page", params={"url": bad_url})
    assert result.ok is False
    assert result.error
    assert created == []


async def test_fetch_content_length_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _FetchStream(b"x", status=200, content_length=str(MAX_FETCH_BYTES + 1))
    _install_fetch(monkeypatch, stream)
    extract_calls = _script_extract(monkeypatch, "never used")
    result = await WebFetchTool().run("need page", params={"url": "https://example.com/big"})
    assert result.ok is False
    assert "larger than" in (result.error or "")
    assert extract_calls == []


async def test_fetch_empty_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _FetchStream(b"<html><body></body></html>", status=200)
    _install_fetch(monkeypatch, stream)
    _script_extract(monkeypatch, None)
    result = await WebFetchTool().run("need page", params={"url": "https://example.com/empty"})
    assert result.ok is False
    assert "no readable text" in (result.error or "")


# ── Follow-ups-only ─────────────────────────────────────────────────────────


async def test_followups_only_web_runs_after_round_zero(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    monkeypatch.setattr(settings, "tool_timeout_s", 5)
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool("radar_search", unique_per_call=True)
    rag = FakeTool("rag_retrieve", unique_per_call=True)
    web = FakeTool("web_search", unique_per_call=True)
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag, "web_search": web})
    original = dispatch_module.run_tool_round
    plans: list[list[tuple[str, str]]] = []
    round_zero_web_calls: list[int] = []

    async def _spy(
        investigation_id: str, planned: list[tuple[str, str]]
    ) -> tuple[int, bool, bool]:
        plans.append(list(planned))
        outcome = await original(investigation_id, planned)
        if len(plans) == 1:
            round_zero_web_calls.append(web.calls)
        return outcome

    monkeypatch.setattr(dispatch_module, "run_tool_round", _spy)
    gap_calls = {"n": 0}

    async def _gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        gap_calls["n"] += 1
        if gap_calls["n"] == 1:
            return GapOutput(
                sufficient=False,
                radar_query="gap-radar-q",
                rag_query="gap-rag-q",
                web_query="gap-web-q",
                rationale="need web",
            )
        return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="done")

    monkeypatch.setattr(workers_module, "assess_gaps", _gap)
    inv = await mgr.create("followups-only probe query", "local")
    try:
        await run_investigation_loop(inv.id)
        assert len(plans) == 2
        assert all(not name.startswith("web_") for name, _query in plans[0])
        assert round_zero_web_calls == [0]
        assert ("web_search", "gap-web-q") in plans[1]
        assert web.calls == 1
        assert web.queries == ["gap-web-q"]
        assert radar.calls == 2
        assert rag.calls == 2
        assert radar.queries[1] == "gap-radar-q"
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.COMPLETE
        assert scripted_milestone == [(0, False), (1, True)]
    finally:
        await mgr.cancel(inv.id)


# ── Cap ─────────────────────────────────────────────────────────────────────


async def test_web_cap_skips_second_followup(
    monkeypatch: pytest.MonkeyPatch, scripted_milestone: list[tuple[int, bool]]
) -> None:
    del scripted_milestone
    monkeypatch.setattr(settings, "max_web_calls", 1)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)
    mgr = _fresh_manager(monkeypatch)
    radar = FakeTool("radar_search", unique_per_call=True)
    rag = FakeTool("rag_retrieve", unique_per_call=True)
    web = FakeTool("web_search", unique_per_call=True)
    _use_registry(monkeypatch, {"radar_search": radar, "rag_retrieve": rag, "web_search": web})
    gap_calls = {"n": 0}

    async def _hungry_gap(board_text: str, query: str) -> GapOutput:
        del board_text, query
        gap_calls["n"] += 1
        if gap_calls["n"] <= 2:
            return GapOutput(
                sufficient=False,
                radar_query=f"radar-q{gap_calls['n']}",
                rag_query=f"rag-q{gap_calls['n']}",
                web_query=f"web-q{gap_calls['n']}",
                rationale="need more",
            )
        return GapOutput(sufficient=True, radar_query="", rag_query="", rationale="done")

    monkeypatch.setattr(workers_module, "assess_gaps", _hungry_gap)
    capped_labels = {"tool": "web_search", "status": "capped"}
    before = REGISTRY.get_sample_value("argus_tool_calls_total", capped_labels) or 0.0
    inv = await mgr.create("web cap probe query", "local")
    try:
        await run_investigation_loop(inv.id)
        assert web.calls == 1
        assert web.queries == ["web-q1"]
        after = REGISTRY.get_sample_value("argus_tool_calls_total", capped_labels) or 0.0
        assert after == before + 1.0
        loaded = await mgr.get(inv.id)
        assert loaded is not None
        assert loaded.usage.web_calls_used == 1
        # Round 0 reserves 2, round 1 reserves 3, round 2 reserves 2 (web capped).
        assert loaded.usage.tool_calls_used == 7
        assert radar.calls == 3
        assert rag.calls == 3
        assert loaded.status == InvestigationStatus.COMPLETE
    finally:
        await mgr.cancel(inv.id)


# ── GapOutput backward compat + budget ──────────────────────────────────────


def test_gap_output_backward_compat() -> None:
    gap = GapOutput(sufficient=True)
    assert gap.web_query == ""
    legacy = GapOutput.model_validate(
        {"sufficient": False, "radar_query": "r", "rag_query": "g", "rationale": "x"}
    )
    assert legacy.web_query == ""
    assert legacy.radar_query == "r"
    assert legacy.rag_query == "g"


async def test_record_web_call_increments_both_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _fresh_manager(monkeypatch)
    inv = await mgr.create("budget probe query", "local")
    try:
        row = await mgr.record_web_call(inv.id)
        assert row is not None
        assert row.usage.tool_calls_used == 1
        assert row.usage.web_calls_used == 1
        row = await mgr.record_web_call(inv.id)
        assert row is not None
        assert row.usage.tool_calls_used == 2
        assert row.usage.web_calls_used == 2
    finally:
        await mgr.cancel(inv.id)
