"""Stage P4-0 - investigation lifecycle (API + manager), mock-only."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient

from app.config import settings
from app.evidence.models import Claim, Evidence, InvestigationStatus, StatusReason
from app.investigations import manager, new_claim_id, new_evidence_id
from app.main import app

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


def _make_evidence(investigation_id: str) -> Evidence:
    return Evidence(
        id=new_evidence_id(),
        investigation_id=investigation_id,
        source_ref="source-1",
        content="key finding content",
        type="text",
        confidence=0.8,
        created_at=time.time(),
    )


def _make_claim(investigation_id: str, evidence_id: str = "") -> Claim:
    return Claim(
        id=new_claim_id(),
        investigation_id=investigation_id,
        statement="core claim statement",
        confidence=0.7,
        evidence_ids=[evidence_id] if evidence_id else [],
    )


# ── Create / read ─────────────────────────────────────────────────────────────


def test_create_defaults_to_local_planned() -> None:
    resp = client.post("/v1/investigate", json={"query": "What caused the outage?"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["investigation_id"].startswith("inv_")
    assert body["user_id"] == "local"
    assert body["status"] == "planned"
    try:
        assert body["status_reason"] is None
    finally:
        client.post(f"/v1/investigate/{body['investigation_id']}/cancel")


def test_explicit_user_id_stored_and_echoed() -> None:
    resp = client.post("/v1/investigate", json={"query": "Whose is this?", "user_id": "alice"})
    assert resp.status_code == 202
    assert resp.json()["user_id"] == "alice"
    inv_id: str = resp.json()["investigation_id"]
    try:
        got = client.get(f"/v1/investigate/{inv_id}")
        assert got.status_code == 200
        assert got.json()["user_id"] == "alice"
        stored = asyncio.run(manager.get(inv_id))
        assert stored is not None
        assert stored.user_id == "alice"  # mandatory in the stored record
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_query_rejected(blank: str) -> None:
    resp = client.post("/v1/investigate", json={"query": blank})
    assert resp.status_code == 422


def test_blank_user_id_rejected() -> None:
    resp = client.post("/v1/investigate", json={"query": "valid query", "user_id": ""})
    assert resp.status_code == 422


def test_get_board_initial_shape() -> None:
    created = client.post("/v1/investigate", json={"query": "shape query"})
    assert created.status_code == 202
    inv_id: str = created.json()["investigation_id"]
    try:
        got = client.get(f"/v1/investigate/{inv_id}")
        assert got.status_code == 200
        body = got.json()
        assert body["schema_version"] == "1.0"
        assert body["evidence"] == []
        assert body["claims"] == []
        assert body["counts"] == {"evidence": 0, "claims": 0}
        assert body["truncated"] is False
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")


def test_get_unknown_returns_404() -> None:
    assert client.get("/v1/investigate/inv_nope_unknown").status_code == 404


# ── Cancel ────────────────────────────────────────────────────────────────────


def test_cancel_is_idempotent_and_visible_on_get() -> None:
    created = client.post("/v1/investigate", json={"query": "cancel me"})
    inv_id: str = created.json()["investigation_id"]
    try:
        first = client.post(f"/v1/investigate/{inv_id}/cancel")
        assert first.status_code == 200
        assert first.json()["status"] == "cancelled"
        assert first.json()["status_reason"] == "cancelled"
        second = client.post(f"/v1/investigate/{inv_id}/cancel")
        assert second.status_code == 200
        assert second.json()["status"] == "cancelled"
        assert second.json()["status_reason"] == "cancelled"
        got = client.get(f"/v1/investigate/{inv_id}")
        assert got.status_code == 200
        assert got.json()["status"] == "cancelled"
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")


def test_cancel_unknown_returns_404() -> None:
    assert client.post("/v1/investigate/inv_nope_unknown/cancel").status_code == 404


# ── Budgets ───────────────────────────────────────────────────────────────────


async def test_tool_call_budget_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "investigation_max_tool_calls", 1)
    inv = await manager.create("tool budget query", "local")
    try:
        updated = await manager.record_tool_call(inv.id)
        assert updated is not None
        assert updated.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert updated.status_reason == StatusReason.TOOL_CALL_LIMIT
    finally:
        await manager.cancel(inv.id)


async def test_iteration_budget_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "investigation_max_iterations", 1)
    inv = await manager.create("iteration budget query", "local")
    try:
        updated = await manager.record_iteration(inv.id)
        assert updated is not None
        assert updated.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert updated.status_reason == StatusReason.ITERATION_LIMIT
    finally:
        await manager.cancel(inv.id)


async def test_wall_clock_budget_check_past_deadline() -> None:
    inv = await manager.create("wall clock query", "local")
    try:
        inv.deadline_at = time.time() - 1.0
        assert manager.check_budgets(inv) == StatusReason.WALL_CLOCK_LIMIT
    finally:
        await manager.cancel(inv.id)


async def test_supervisor_expires_past_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "investigation_max_wall_time_s", 1)
    inv = await manager.create("supervisor expiry query", "local")
    try:
        await asyncio.sleep(1.4)
        loaded = await manager.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert loaded.status_reason == StatusReason.WALL_CLOCK_LIMIT
    finally:
        await manager.cancel(inv.id)


# ── Fail-open / TTL / store ───────────────────────────────────────────────────


async def test_fail_open_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", None)
    inv = await manager.create("fail open query", "local")
    try:
        got = await manager.get(inv.id)
        assert got is not None
        assert got.id == inv.id
        cancelled = await manager.cancel(inv.id)
        assert cancelled is not None
        assert cancelled.status == InvestigationStatus.CANCELLED
    finally:
        await manager.cancel(inv.id)


async def test_redis_keys_carry_ttl(fake_redis: Any) -> None:
    inv = await manager.create("ttl query", "local")
    try:
        for suffix in ("meta", "evidence", "claims"):
            ttl: int = await fake_redis.ttl(f"argus:inv:{inv.id}:{suffix}")
            assert 0 < ttl <= settings.investigation_ttl_s
    finally:
        await manager.cancel(inv.id)


async def test_add_evidence_and_claim_round_trip() -> None:
    inv = await manager.create("round trip query", "local")
    try:
        ev = _make_evidence(inv.id)
        assert await manager.add_evidence(inv.id, ev) is not None
        assert await manager.add_claim(inv.id, _make_claim(inv.id, ev.id)) is not None
        resp = client.get(f"/v1/investigate/{inv.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["counts"] == {"evidence": 1, "claims": 1}
        assert body["evidence"][0]["content"] == "key finding content"
        assert body["claims"][0]["statement"] == "core claim statement"
    finally:
        await manager.cancel(inv.id)


async def test_add_to_terminal_investigation_raises() -> None:
    inv = await manager.create("terminal add query", "local")
    try:
        assert await manager.cancel(inv.id) is not None
        with pytest.raises(ValueError):
            await manager.add_evidence(inv.id, _make_evidence(inv.id))
        with pytest.raises(ValueError):
            await manager.add_claim(inv.id, _make_claim(inv.id))
    finally:
        await manager.cancel(inv.id)


async def test_illegal_transition_raises() -> None:
    inv = await manager.create("illegal transition query", "local")
    try:
        with pytest.raises(ValueError):
            await manager.transition(inv.id, InvestigationStatus.COMPLETE)
    finally:
        await manager.cancel(inv.id)


# ── No authorization branching ────────────────────────────────────────────────


def test_read_and_cancel_ignore_user_id() -> None:
    alice = client.post("/v1/investigate", json={"query": "alice query", "user_id": "alice"})
    local = client.post("/v1/investigate", json={"query": "local query"})
    assert alice.status_code == 202
    assert local.status_code == 202
    alice_id: str = alice.json()["investigation_id"]
    local_id: str = local.json()["investigation_id"]
    try:
        # Cross-identity reads succeed; "local" grants nothing special, no 403/401 anywhere.
        for inv_id in (alice_id, local_id):
            assert client.get(f"/v1/investigate/{inv_id}").status_code == 200
        cancelled_alice = client.post(f"/v1/investigate/{alice_id}/cancel")
        cancelled_local = client.post(f"/v1/investigate/{local_id}/cancel")
        assert cancelled_alice.status_code == 200
        assert cancelled_local.status_code == 200
        assert cancelled_alice.json()["status"] == cancelled_local.json()["status"] == "cancelled"
    finally:
        client.post(f"/v1/investigate/{alice_id}/cancel")
        client.post(f"/v1/investigate/{local_id}/cancel")
