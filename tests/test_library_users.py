"""P7-2 document-library proxy per-user (mock-only, no live RAG calls).

Uses plain TestClient(app) WITHOUT lifespan (lifespan would overwrite
holder.client — see tests/test_multitenant_isolation.py pattern), fakeredis
holder injection, and monkeypatched singleton methods + httpx.AsyncClient.
"""

from typing import Any

import httpx
import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient

from app.auth import create_access_token
from app.config import settings
from app.main import app
from app.rediskit import holder
from app.tools.rag import rag_retrieve_tool

DOCS_PAYLOAD = {"count": 1, "documents": [{"name": "report.pdf", "size_bytes": 1024}]}


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
    """Stand-in for httpx.AsyncClient; serves staged proxy GETs only."""

    instances: list["FakeAsyncClient"] = []
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
    default_payload: Any = None,
) -> None:
    FakeAsyncClient.instances = []
    FakeAsyncClient.get_script = list(get_script) if get_script else []
    FakeAsyncClient.default_payload = default_payload if default_payload is not None else {}


def _get_calls() -> list[dict[str, Any]]:
    return [c for inst in FakeAsyncClient.instances for c in inst.calls if c["method"] == "GET"]


@pytest.fixture
def library_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Plain TestClient (no lifespan) with fakeredis holder + library enabled."""
    _reset_fake()
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(holder, "client", fr)
    monkeypatch.setattr(holder, "cache", None)
    monkeypatch.setattr(settings, "workspace_rag_enabled", True)
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.test")
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    return TestClient(app)


def _auth_headers(subject: str) -> dict[str, str]:
    token, _ = create_access_token(subject)
    return {"Authorization": f"Bearer {token}"}


def test_owner_flows_into_token_call(
    library_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "users-test-secret")
    monkeypatch.setattr(settings, "auth_single_user_mode", True)
    _reset_fake(get_script=[FakeUpstreamResponse(payload=DOCS_PAYLOAD)])

    ensure_calls: list[str] = []

    async def fake_ensure(owner: str = "local") -> str:
        ensure_calls.append(owner)
        return f"tok-for-{owner}"

    monkeypatch.setattr(rag_retrieve_tool, "_ensure_token", fake_ensure)
    monkeypatch.setattr(
        rag_retrieve_tool, "_invalidate_subject", lambda owner="local": None
    )

    resp = library_client.get("/v1/library/documents", headers=_auth_headers("alice"))
    assert resp.status_code == 200
    assert resp.json() == DOCS_PAYLOAD
    assert ensure_calls == ["alice"]
    gets = _get_calls()
    assert len(gets) == 1
    assert gets[0]["headers"] == {"Authorization": "Bearer tok-for-alice"}


def test_per_subject_401_retry_invalidates_alice(
    library_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "users-test-secret")
    monkeypatch.setattr(settings, "auth_single_user_mode", True)
    _reset_fake(
        get_script=[
            FakeUpstreamResponse(status_code=401, payload={"detail": "expired"}),
            FakeUpstreamResponse(payload=DOCS_PAYLOAD),
        ]
    )

    ensure_calls: list[str] = []
    invalidate_calls: list[str] = []
    tokens = ["tok-old", "tok-new"]

    async def fake_ensure(owner: str = "local") -> str:
        ensure_calls.append(owner)
        return tokens[min(len(ensure_calls) - 1, len(tokens) - 1)]

    def fake_invalidate(owner: str = "local") -> None:
        invalidate_calls.append(owner)

    monkeypatch.setattr(rag_retrieve_tool, "_ensure_token", fake_ensure)
    monkeypatch.setattr(rag_retrieve_tool, "_invalidate_subject", fake_invalidate)

    resp = library_client.get("/v1/library/documents", headers=_auth_headers("alice"))
    assert resp.status_code == 200
    assert resp.json() == DOCS_PAYLOAD
    assert ensure_calls == ["alice", "alice"]
    assert invalidate_calls == ["alice"]
    gets = _get_calls()
    assert len(gets) == 2
    assert gets[0]["headers"] == {"Authorization": "Bearer tok-old"}
    assert gets[1]["headers"] == {"Authorization": "Bearer tok-new"}


def test_anonymous_single_user_resolves_local(
    library_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "auth_single_user_mode", True)
    _reset_fake(get_script=[FakeUpstreamResponse(payload=DOCS_PAYLOAD)])

    ensure_calls: list[str] = []

    async def fake_ensure(owner: str = "local") -> str:
        ensure_calls.append(owner)
        return f"tok-for-{owner}"

    monkeypatch.setattr(rag_retrieve_tool, "_ensure_token", fake_ensure)
    monkeypatch.setattr(
        rag_retrieve_tool, "_invalidate_subject", lambda owner="local": None
    )

    resp = library_client.get("/v1/library/documents")
    assert resp.status_code == 200
    assert ensure_calls == ["local"]
    assert _get_calls()[0]["headers"] == {"Authorization": "Bearer tok-for-local"}


@pytest.mark.parametrize(
    ("upstream_status", "expected_status"),
    [(429, 429), (502, 502)],
)
def test_upstream_mapping_unchanged(
    library_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    upstream_status: int,
    expected_status: int,
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "auth_single_user_mode", True)
    _reset_fake(
        get_script=[FakeUpstreamResponse(status_code=upstream_status, payload={"detail": "x"})]
    )

    async def fake_ensure(owner: str = "local") -> str:
        return "tok"

    monkeypatch.setattr(rag_retrieve_tool, "_ensure_token", fake_ensure)
    monkeypatch.setattr(
        rag_retrieve_tool, "_invalidate_subject", lambda owner="local": None
    )

    resp = library_client.get("/v1/library/collections")
    assert resp.status_code == expected_status


def test_disabled_flag_still_404(
    library_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "workspace_rag_enabled", False)
    monkeypatch.setattr(settings, "auth_enabled", False)
    _reset_fake()
    resp = library_client.get("/v1/library/documents")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Document library is disabled."}
    assert FakeAsyncClient.instances == []
