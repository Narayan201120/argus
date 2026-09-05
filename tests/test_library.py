"""P4-4 RAG Library proxy routes (mock-only, no live RAG calls)."""

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.api.routes import library
from app.config import settings
from app.tools.rag import rag_retrieve_tool

library_app = FastAPI()
library_app.include_router(library.router, prefix="/v1")
client = TestClient(library_app)

DOCS_PAYLOAD = {
    "count": 2,
    "documents": [
        {"name": "report.pdf", "size_bytes": 1024},
        {"name": "notes.txt", "size_bytes": 256},
    ],
}
DOC_PAYLOAD = {
    "name": "report.pdf",
    "extension": ".pdf",
    "content": "first page text",
    "total_characters": 15000,
    "truncated": False,
}
COLLECTIONS_PAYLOAD = {
    "count": 1,
    "collections": [
        {
            "id": "c1",
            "name": "Research",
            "description": "papers",
            "document_count": 2,
            "created_at": "2024-01-01",
        }
    ],
}
COLLECTION_PAYLOAD = {
    "id": "c1",
    "name": "Research",
    "description": "papers",
    "document_count": 2,
    "created_at": "2024-01-01",
    "documents": [{"name": "report.pdf", "size_bytes": 1024}],
}


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
    """Hand-rolled stand-in for httpx.AsyncClient; serves sign-in POSTs and proxy GETs.

    Sign-in tokens are staged via signin_tokens (popped per POST, last one reused);
    proxy GET outcomes are staged in order via get_script (response or exception).
    """

    instances: list["FakeAsyncClient"] = []
    signin_tokens: list[str] = ["test-token"]
    get_script: list[Any] = []
    default_payload: Any = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        FakeAsyncClient.instances.append(self)

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeUpstreamResponse:
        self.calls.append({"method": "POST", "url": url, "json": json, "headers": headers})
        if len(FakeAsyncClient.signin_tokens) > 1:
            token = FakeAsyncClient.signin_tokens.pop(0)
        else:
            token = FakeAsyncClient.signin_tokens[0]
        return FakeUpstreamResponse(payload={"message": "ok", "tokens": {"access": token}})

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeUpstreamResponse:
        self.calls.append({"method": "GET", "url": url, "params": params, "headers": headers})
        if FakeAsyncClient.get_script:
            item = FakeAsyncClient.get_script.pop(0)
        else:
            item = FakeUpstreamResponse(payload=FakeAsyncClient.default_payload)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, FakeUpstreamResponse)
        return item


def _reset_fake(
    *,
    get_script: list[Any] | None = None,
    signin_tokens: list[str] | None = None,
    default_payload: Any = None,
) -> None:
    FakeAsyncClient.instances = []
    FakeAsyncClient.get_script = list(get_script) if get_script else []
    FakeAsyncClient.signin_tokens = list(signin_tokens) if signin_tokens else ["test-token"]
    FakeAsyncClient.default_payload = default_payload if default_payload is not None else {}


@pytest.fixture
def library_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    monkeypatch.setattr(settings, "workspace_rag_enabled", True)
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.test")
    monkeypatch.setattr(settings, "rag_service_user", "svc-user")
    monkeypatch.setattr(settings, "rag_service_pass", "svc-pass")
    # The shared tool singleton caches its token; reset so each test signs in fresh.
    monkeypatch.setattr(rag_retrieve_tool, "_access_token", None)
    monkeypatch.setattr(rag_retrieve_tool, "_token_expires_at", 0.0)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)


def _get_calls() -> list[dict[str, Any]]:
    return [c for inst in FakeAsyncClient.instances for c in inst.calls if c["method"] == "GET"]


def _post_calls() -> list[dict[str, Any]]:
    return [c for inst in FakeAsyncClient.instances for c in inst.calls if c["method"] == "POST"]


def _counter(status: str) -> float:
    return REGISTRY.get_sample_value(
        "argus_proxy_calls_total", {"service": "rag", "status": status}
    ) or 0.0


def test_disabled_documents_returns_404_with_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    monkeypatch.setattr(settings, "workspace_rag_enabled", False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    before = _counter("disabled")
    response = client.get("/v1/library/documents")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document library is disabled."}
    assert _counter("disabled") == before + 1
    assert FakeAsyncClient.instances == []  # no upstream client constructed


def test_disabled_detail_and_collection_routes_return_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    monkeypatch.setattr(settings, "workspace_rag_enabled", False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    assert client.get("/v1/library/documents/report.pdf").status_code == 404
    assert client.get("/v1/library/collections").status_code == 404
    assert client.get("/v1/library/collections/c1").status_code == 404
    assert FakeAsyncClient.instances == []


def test_list_documents_forwards_url_and_bearer(library_enabled: None) -> None:
    _reset_fake(
        get_script=[FakeUpstreamResponse(payload=DOCS_PAYLOAD)],
        signin_tokens=["tok-abc"],
    )
    response = client.get("/v1/library/documents")
    assert response.status_code == 200
    assert response.json() == DOCS_PAYLOAD  # body returned unchanged
    gets = _get_calls()
    assert len(gets) == 1
    assert gets[0]["url"] == "http://rag.test/api/documents/"
    assert gets[0]["params"] is None
    assert gets[0]["headers"] == {"Authorization": "Bearer tok-abc"}
    posts = _post_calls()
    assert len(posts) == 1
    assert posts[0]["url"] == "http://rag.test/api/sign-in/"


def test_document_detail_forwards_filename(library_enabled: None) -> None:
    _reset_fake(
        get_script=[FakeUpstreamResponse(payload=DOC_PAYLOAD)],
        signin_tokens=["tok-abc"],
    )
    response = client.get("/v1/library/documents/report.pdf")
    assert response.status_code == 200
    assert response.json() == DOC_PAYLOAD
    assert _get_calls()[-1]["url"] == "http://rag.test/api/documents/report.pdf/"


def test_collections_list_and_detail_urls(library_enabled: None) -> None:
    _reset_fake(
        get_script=[FakeUpstreamResponse(payload=COLLECTIONS_PAYLOAD)],
        signin_tokens=["tok-abc"],
    )
    response = client.get("/v1/library/collections")
    assert response.status_code == 200
    assert response.json() == COLLECTIONS_PAYLOAD
    assert _get_calls()[-1]["url"] == "http://rag.test/api/collections/"
    _reset_fake(
        get_script=[FakeUpstreamResponse(payload=COLLECTION_PAYLOAD)],
        signin_tokens=["tok-abc"],
    )
    response = client.get("/v1/library/collections/c1")
    assert response.status_code == 200
    assert response.json() == COLLECTION_PAYLOAD
    assert _get_calls()[-1]["url"] == "http://rag.test/api/collections/c1/"


def test_timeout_forwarded_to_httpx(library_enabled: None) -> None:
    _reset_fake(get_script=[FakeUpstreamResponse(payload=DOCS_PAYLOAD)])
    assert client.get("/v1/library/documents").status_code == 200
    assert all(inst.init_kwargs.get("timeout") == settings.tool_timeout_s for inst in FakeAsyncClient.instances)


def test_401_refresh_retry_then_success(library_enabled: None) -> None:
    _reset_fake(
        get_script=[
            FakeUpstreamResponse(status_code=401, payload={"detail": "expired"}),
            FakeUpstreamResponse(payload=DOCS_PAYLOAD),
        ],
        signin_tokens=["tok-old", "tok-new"],
    )
    response = client.get("/v1/library/documents")
    assert response.status_code == 200
    assert response.json() == DOCS_PAYLOAD
    gets = _get_calls()
    assert len(gets) == 2
    assert gets[0]["headers"] == {"Authorization": "Bearer tok-old"}
    assert gets[1]["headers"] == {"Authorization": "Bearer tok-new"}
    assert len(_post_calls()) == 2  # initial sign-in plus one refresh


def test_persistent_401_returns_502(library_enabled: None) -> None:
    _reset_fake(
        get_script=[
            FakeUpstreamResponse(status_code=401, payload={"detail": "expired"}),
            FakeUpstreamResponse(status_code=401, payload={"detail": "still bad"}),
        ]
    )
    before = _counter("error")
    response = client.get("/v1/library/documents")
    assert response.status_code == 502
    assert _counter("error") == before + 1


def test_429_forwards_retry_after(library_enabled: None) -> None:
    _reset_fake(
        get_script=[
            FakeUpstreamResponse(
                status_code=429, payload={"detail": "slow down"}, headers={"Retry-After": "7"}
            )
        ]
    )
    before = _counter("error")
    response = client.get("/v1/library/collections")
    assert response.status_code == 429
    assert response.headers.get("retry-after") == "7"
    assert _counter("error") == before + 1


def test_502_on_upstream_500(library_enabled: None) -> None:
    _reset_fake(get_script=[FakeUpstreamResponse(status_code=500, payload={"detail": "boom"})])
    before = _counter("error")
    response = client.get("/v1/library/documents")
    assert response.status_code == 502
    assert _counter("error") == before + 1


def test_502_on_timeout(library_enabled: None) -> None:
    _reset_fake(get_script=[httpx.ConnectTimeout("timed out")])
    before = _counter("timeout")
    response = client.get("/v1/library/documents/report.pdf")
    assert response.status_code == 502
    assert _counter("timeout") == before + 1


def test_502_on_invalid_json(library_enabled: None) -> None:
    _reset_fake(get_script=[FakeUpstreamResponse(payload=ValueError("no json"))])
    assert client.get("/v1/library/documents").status_code == 502


def test_ok_counter_increments(library_enabled: None) -> None:
    _reset_fake(get_script=[FakeUpstreamResponse(payload=DOCS_PAYLOAD)])
    before = _counter("ok")
    assert client.get("/v1/library/documents").status_code == 200
    assert _counter("ok") == before + 1
