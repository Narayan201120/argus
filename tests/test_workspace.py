"""P4-4 Research Radar workspace proxy routes (mock-only, no live Radar calls)."""

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.api.routes import workspace
from app.config import settings

workspace_app = FastAPI()
workspace_app.include_router(workspace.router, prefix="/v1")
client = TestClient(workspace_app)

LIST_PAYLOAD = {
    "items": [
        {
            "id": "p1",
            "title": "Attention Is All You Need",
            "publication_year": 2017,
            "cited_by_count": 90000,
            "authors": [{"id": "a1", "name": "Vaswani"}],
        }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20,
}
DETAIL_PAYLOAD = {
    "id": "p1",
    "title": "Attention Is All You Need",
    "abstract": "We propose the Transformer.",
    "publication_year": 2017,
    "doi": "10.1000/xyz",
    "cited_by_count": 90000,
    "created_at": "2024-01-01",
    "authors": [],
    "topics": [],
}
SIMILAR_PAYLOAD = [{"id": "p2", "title": "BERT", "similarity_score": 0.9123}]


class FakeUpstreamResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = httpx.Headers(headers or {})
        self.text = str(payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeAsyncClient:
    """Hand-rolled stand-in for httpx.AsyncClient; records constructor and GET calls.

    The route builds one client per request, so scripted behavior lives on
    class attributes staged via _stage() before each request.
    """

    next_response: FakeUpstreamResponse = FakeUpstreamResponse()
    next_exc: BaseException | None = None
    last_instance: "FakeAsyncClient | None" = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.response = FakeAsyncClient.next_response
        self.exc = FakeAsyncClient.next_exc
        FakeAsyncClient.last_instance = self

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False

    async def get(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> FakeUpstreamResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        if self.exc is not None:
            raise self.exc
        return self.response


def _stage(
    response: FakeUpstreamResponse | None = None, exc: BaseException | None = None
) -> None:
    FakeAsyncClient.next_response = response or FakeUpstreamResponse()
    FakeAsyncClient.next_exc = exc


@pytest.fixture
def radar_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncClient.last_instance = None
    _stage()
    monkeypatch.setattr(settings, "workspace_radar_enabled", True)
    monkeypatch.setattr(settings, "radar_base_url", "http://radar.test")
    monkeypatch.setattr(settings, "radar_api_key", "")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)


def _used_fake() -> FakeAsyncClient:
    assert FakeAsyncClient.last_instance is not None
    return FakeAsyncClient.last_instance


def _counter(status: str) -> float:
    return REGISTRY.get_sample_value(
        "argus_proxy_calls_total", {"service": "radar", "status": status}
    ) or 0.0


def test_disabled_list_returns_404(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncClient.last_instance = None
    monkeypatch.setattr(settings, "workspace_radar_enabled", False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    before = _counter("disabled")
    response = client.get("/v1/radar/papers", params={"q": "transformer"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Radar workspace is disabled."}
    assert _counter("disabled") == before + 1
    assert FakeAsyncClient.last_instance is None  # no upstream client constructed


def test_disabled_detail_and_similar_return_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "workspace_radar_enabled", False)
    assert client.get("/v1/radar/papers/p1").status_code == 404
    assert client.get("/v1/radar/papers/p1/similar").status_code == 404


def test_list_forwards_params_and_caps_page_size(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(payload=LIST_PAYLOAD))
    response = client.get(
        "/v1/radar/papers",
        params={"q": "transformer", "year": 2017, "topic": "nlp",
                "author": "vas", "page": 2, "page_size": 200},
    )
    assert response.status_code == 200
    assert response.json() == LIST_PAYLOAD  # body returned unchanged
    call = _used_fake().calls[-1]
    assert call["url"] == "http://radar.test/papers"
    assert call["params"] == {
        "q": "transformer", "year": 2017, "topic": "nlp",
        "author": "vas", "page": 2, "page_size": 50,
    }
    assert call["headers"] == {}  # no key configured -> none sent


def test_list_omits_unset_optional_params(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(payload=LIST_PAYLOAD))
    assert client.get("/v1/radar/papers").status_code == 200
    assert _used_fake().calls[-1]["params"] == {"page": 1, "page_size": 20}


def test_api_key_sent_only_when_configured(
    radar_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(FakeUpstreamResponse(payload=LIST_PAYLOAD))
    client.get("/v1/radar/papers")
    assert _used_fake().calls[-1]["headers"] == {}
    monkeypatch.setattr(settings, "radar_api_key", "secret-key")
    client.get("/v1/radar/papers")
    assert _used_fake().calls[-1]["headers"] == {"X-API-Key": "secret-key"}


def test_timeout_forwarded_to_httpx(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(payload=LIST_PAYLOAD))
    client.get("/v1/radar/papers")
    assert _used_fake().init_kwargs.get("timeout") == settings.tool_timeout_s


def test_detail_and_similar_urls_and_no_params(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(payload=DETAIL_PAYLOAD))
    response = client.get("/v1/radar/papers/p1")
    assert response.status_code == 200
    assert response.json() == DETAIL_PAYLOAD
    assert _used_fake().calls[-1] == {
        "url": "http://radar.test/papers/p1", "params": None, "headers": {},
    }
    _stage(FakeUpstreamResponse(payload=SIMILAR_PAYLOAD))
    response = client.get("/v1/radar/papers/p1/similar")
    assert response.status_code == 200
    assert response.json() == SIMILAR_PAYLOAD
    assert _used_fake().calls[-1]["url"] == "http://radar.test/papers/p1/similar"
    assert _used_fake().calls[-1]["params"] is None


def test_429_forwards_retry_after(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(
        status_code=429, payload={"detail": "slow down"}, headers={"Retry-After": "7"}
    ))
    before = _counter("error")
    response = client.get("/v1/radar/papers")
    assert response.status_code == 429
    assert response.headers.get("retry-after") == "7"
    assert _counter("error") == before + 1


def test_429_without_retry_after_still_429(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(status_code=429, payload={"detail": "slow"}))
    response = client.get("/v1/radar/papers/p1")
    assert response.status_code == 429
    assert "retry-after" not in {k.lower() for k in response.headers}


def test_502_on_upstream_500(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(status_code=500, payload={"detail": "boom"}))
    before = _counter("error")
    response = client.get("/v1/radar/papers")
    assert response.status_code == 502
    assert _counter("error") == before + 1


def test_502_on_timeout(radar_enabled: None) -> None:
    _stage(exc=httpx.ConnectTimeout("timed out"))
    before = _counter("timeout")
    response = client.get("/v1/radar/papers/p1/similar")
    assert response.status_code == 502
    assert _counter("timeout") == before + 1


def test_502_on_invalid_json(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(payload=ValueError("no json")))
    assert client.get("/v1/radar/papers").status_code == 502


def test_ok_counter_increments(radar_enabled: None) -> None:
    _stage(FakeUpstreamResponse(payload=LIST_PAYLOAD))
    before = _counter("ok")
    assert client.get("/v1/radar/papers").status_code == 200
    assert _counter("ok") == before + 1
