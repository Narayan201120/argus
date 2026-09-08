"""P7-1 isolation matrix (DEC-056). Mock-only, fakeredis.

Authenticated identities are minted directly via create_access_token so
no client-credential exchange is needed. Every cross-user access must be
404 (never 403 or 401): no id oracle. Missing credentials stay 401 from
the middleware.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient

from app.auth import create_access_token
from app.config import settings
from app.investigations import manager
from app.main import app


@pytest.fixture
async def fake_redis() -> AsyncIterator[Any]:
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield fr
    await fr.aclose()


@pytest.fixture
def authed_client(
    fake_redis: Any, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    from app.rediskit import holder

    # Plain TestClient, no lifespan: entering the lifespan would overwrite
    # holder.client with a real-Redis connect attempt. The module-level
    # pattern in test_investigations.py works the same way.
    monkeypatch.setattr(holder, "client", fake_redis)
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "iso-test-secret")
    monkeypatch.setattr(settings, "auth_require_auth", True)
    return TestClient(app)


def _headers(subject: str) -> dict[str, str]:
    token, _ = create_access_token(subject)
    return {"Authorization": f"Bearer {token}"}


async def _make_owned(query: str, owner: str) -> str:
    inv = await manager.create(query, owner)
    return inv.id


async def _cleanup(inv_id: str) -> None:
    try:
        await manager.cancel(inv_id)
    except Exception:  # noqa: BLE001 - best effort
        pass


def test_create_ignores_client_user_id(
    authed_client: TestClient, scripted_milestone: list[tuple[int, bool]]
) -> None:
    resp = authed_client.post(
        "/v1/investigate",
        json={"query": "spoofed owner probe", "user_id": "bob"},
        headers=_headers("alice"),
    )
    assert resp.status_code == 202
    assert resp.json()["user_id"] == "alice"
    authed_client.post(
        f"/v1/investigate/{resp.json()['investigation_id']}/cancel",
        headers=_headers("alice"),
    )


def test_owner_full_lifecycle(
    authed_client: TestClient, scripted_milestone: list[tuple[int, bool]]
) -> None:
    created = authed_client.post(
        "/v1/investigate",
        json={"query": "owner lifecycle probe"},
        headers=_headers("alice"),
    )
    assert created.status_code == 202
    inv_id: str = created.json()["investigation_id"]
    try:
        assert (
            authed_client.get(
                f"/v1/investigate/{inv_id}", headers=_headers("alice")
            ).status_code
            == 200
        )
        rated = authed_client.post(
            f"/v1/investigate/{inv_id}/feedback",
            json={"rating": 4},
            headers=_headers("alice"),
        )
        assert rated.status_code == 200
        cancelled = authed_client.post(
            f"/v1/investigate/{inv_id}/cancel", headers=_headers("alice")
        )
        assert cancelled.status_code == 200
    finally:
        authed_client.post(
            f"/v1/investigate/{inv_id}/cancel", headers=_headers("alice")
        )


@pytest.mark.asyncio
async def test_cross_user_access_is_404(
    authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    inv_id = await _make_owned("cross user probe", "alice")
    try:
        assert (
            authed_client.get(
                f"/v1/investigate/{inv_id}", headers=_headers("bob")
            ).status_code
            == 404
        )
        assert (
            authed_client.post(
                f"/v1/investigate/{inv_id}/cancel", headers=_headers("bob")
            ).status_code
            == 404
        )
        assert (
            authed_client.post(
                f"/v1/investigate/{inv_id}/feedback",
                json={"rating": 5},
                headers=_headers("bob"),
            ).status_code
            == 404
        )
        assert (
            authed_client.get(
                f"/v1/investigate/{inv_id}/stream", headers=_headers("bob")
            ).status_code
            == 404
        )
        listed = authed_client.get(
            "/v1/investigations", params={"limit": 100}, headers=_headers("bob")
        )
        assert listed.status_code == 200
        assert all(
            s["investigation_id"] != inv_id
            for s in listed.json()["investigations"]
        )
    finally:
        await _cleanup(inv_id)


def test_unauthenticated_is_401(authed_client: TestClient) -> None:
    assert authed_client.post("/v1/investigate", json={"query": "x"}).status_code == 401
    assert authed_client.get("/v1/investigate/inv_nope").status_code == 401
    assert authed_client.get("/v1/investigations").status_code == 401
    assert authed_client.post("/v1/investigate/inv_nope/cancel").status_code == 401
    assert (
        authed_client.post(
            "/v1/investigate/inv_nope/feedback", json={"rating": 3}
        ).status_code
        == 401
    )


@pytest.mark.asyncio
async def test_legacy_local_invisible_to_named_subject(
    authed_client: TestClient,
) -> None:
    inv_id = await _make_owned("legacy local probe", "local")
    try:
        assert (
            authed_client.get(
                f"/v1/investigate/{inv_id}", headers=_headers("alice")
            ).status_code
            == 404
        )
    finally:
        await _cleanup(inv_id)


@pytest.mark.asyncio
async def test_session_memory_isolated_by_owner(
    authed_client: TestClient,
) -> None:
    from app.memory import session_store

    await session_store.append("iso-sess", "q", "a", owner="alice")
    try:
        bob = authed_client.get("/v1/session/iso-sess", headers=_headers("bob"))
        assert bob.status_code == 200
        assert bob.json()["turns"] == []
        alice = authed_client.get(
            "/v1/session/iso-sess", headers=_headers("alice")
        )
        assert alice.status_code == 200
        assert len(alice.json()["turns"]) == 1
    finally:
        await session_store.clear("iso-sess", owner="alice")


@pytest.mark.asyncio
async def test_feedback_isolated_by_owner(authed_client: TestClient) -> None:
    from app.feedback import save_rating

    assert await save_rating("iso-req-12345678", 5, owner="alice") is True
    assert (
        authed_client.get(
            "/v1/feedback/iso-req-12345678", headers=_headers("bob")
        ).status_code
        == 404
    )
    assert (
        authed_client.get(
            "/v1/feedback/iso-req-12345678", headers=_headers("alice")
        ).status_code
        == 200
    )


def test_cache_payload_binds_owner() -> None:
    from app.api.routes.query import _cache_payload
    from app.api.schemas import QueryRequest

    request = QueryRequest(query="same question")
    assert _cache_payload(request, "alice") != _cache_payload(request, "bob")
    assert _cache_payload(request, "alice") == _cache_payload(request, "alice")


@pytest.mark.asyncio
async def test_report_jobs_scoped_and_legacy_gated(
    authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.orchestration.report_jobs import report_job_store

    job = await report_job_store.create("owned report probe", owner="alice")
    try:
        assert (
            authed_client.get(
                f"/v1/report/{job.job_id}", headers=_headers("alice")
            ).status_code
            == 200
        )
        assert (
            authed_client.get(
                f"/v1/report/{job.job_id}", headers=_headers("bob")
            ).status_code
            == 404
        )
    finally:
        report_job_store._jobs.pop(job.job_id, None)

    legacy = await report_job_store.create("legacy report probe")
    try:
        assert (
            authed_client.get(
                f"/v1/report/{legacy.job_id}", headers=_headers("alice")
            ).status_code
            == 200
        )
        monkeypatch.setattr(settings, "auth_single_user_mode", False)
        assert (
            authed_client.get(
                f"/v1/report/{legacy.job_id}", headers=_headers("alice")
            ).status_code
            == 404
        )
    finally:
        monkeypatch.undo()
        report_job_store._jobs.pop(legacy.job_id, None)
