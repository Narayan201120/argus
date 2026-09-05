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


async def test_cancel_is_idempotent_and_visible_on_get() -> None:
    # Created at manager level: no background loop races the cancel.
    inv = await manager.create("cancel me", "local")
    inv_id = inv.id
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
        first = await manager.record_tool_call(inv.id)
        assert first is not None
        assert first.status == InvestigationStatus.PLANNED  # budget of 1 allows 1 call
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
        first = await manager.record_iteration(inv.id)
        assert first is not None
        assert first.status == InvestigationStatus.PLANNED  # budget of 1 allows 1 round
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


async def test_read_and_cancel_ignore_user_id() -> None:
    # Manager-level creation keeps the rows pre-loop so cancels are deterministic.
    alice_inv = await manager.create("alice query", "alice")
    local_inv = await manager.create("local query", "local")
    alice_id = alice_inv.id
    local_id = local_inv.id
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


# ── List recent (P4-4) ────────────────────────────────────────────────────────


async def test_list_recent_ordering_newest_first() -> None:
    from app.investigations import InvestigationManager

    fresh = InvestigationManager()
    ids: list[str] = []
    try:
        for label in ("first query", "second query", "third query"):
            inv = await fresh.create(label, "local")
            ids.append(inv.id)
            await asyncio.sleep(0.05)  # exceed Windows ~15ms clock granularity
        recent = await fresh.list_recent(20)
        got = [inv.id for inv in recent[:3]]
        assert got == [ids[2], ids[1], ids[0]]
    finally:
        for inv_id in ids:
            await fresh.cancel(inv_id)


async def test_list_recent_limit_cap_respected() -> None:
    from app.investigations import InvestigationManager

    fresh = InvestigationManager()
    ids: list[str] = []
    try:
        for n in range(3):
            inv = await fresh.create(f"cap query {n}", "local")
            ids.append(inv.id)
            await asyncio.sleep(0.05)  # exceed Windows ~15ms clock granularity
        assert len(await fresh.list_recent(2)) == 2
        assert [inv.id for inv in await fresh.list_recent(2)] == [ids[2], ids[1]]
        oversized = await fresh.list_recent(1000)
        assert len(oversized) == 3  # clamped to 100, not an error
        assert len(await fresh.list_recent(0)) == 1  # clamped to 1
    finally:
        for inv_id in ids:
            await fresh.cancel(inv_id)


async def test_list_recent_empty_store_returns_empty() -> None:
    from app.evidence.store import EvidenceBoardStore
    from app.investigations import InvestigationManager

    assert await EvidenceBoardStore().list_recent(20) == []
    assert await InvestigationManager(EvidenceBoardStore()).list_recent(20) == []


async def test_list_recent_fail_open_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.investigations import InvestigationManager
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", None)
    fresh = InvestigationManager()
    first = await fresh.create("no redis one", "local")
    await asyncio.sleep(0.05)  # exceed Windows ~15ms clock granularity for ordering
    second = await fresh.create("no redis two", "local")
    try:
        recent = await fresh.list_recent(20)
        assert [inv.id for inv in recent[:2]] == [second.id, first.id]
    finally:
        await fresh.cancel(first.id)
        await fresh.cancel(second.id)


async def test_list_summaries_carry_correct_counts() -> None:
    from app.analysis.synthesis import SynthesisRecord, synthesis_store

    long_query = "q" * 300
    created = await manager.create(long_query, "alice")
    inv_id = created.id
    try:
        ev1 = _make_evidence(inv_id)
        ev2 = _make_evidence(inv_id)
        assert await manager.add_evidence(inv_id, ev1) is not None
        assert await manager.add_evidence(inv_id, ev2) is not None
        assert await manager.add_claim(inv_id, _make_claim(inv_id, ev1.id)) is not None
        records = [
            SynthesisRecord(milestone=0, markdown="report one", final=False, created_at=time.time()),
            SynthesisRecord(milestone=1, markdown="report two", final=True, created_at=time.time()),
        ]
        await synthesis_store.save_all(inv_id, records, settings.investigation_ttl_s)
        resp = client.get("/v1/investigations", params={"limit": 100})
        assert resp.status_code == 200
        summaries = resp.json()["investigations"]
        match = next(s for s in summaries if s["investigation_id"] == inv_id)
        assert match["user_id"] == "alice"
        assert match["query"] == "q" * 200  # truncated to 200 chars in the route
        assert match["evidence_count"] == 2
        assert match["claim_count"] == 1
        assert match["synthesis_count"] == 2
        assert match["status"] == "planned"
        assert match["created_at"] <= match["updated_at"]
    finally:
        synthesis_store._memory.pop(inv_id, None)
        await manager.cancel(inv_id)


def test_list_route_limit_param_respected() -> None:
    created = client.post("/v1/investigate", json={"query": "limit param query"})
    assert created.status_code == 202
    inv_id: str = created.json()["investigation_id"]
    try:
        resp = client.get("/v1/investigations", params={"limit": 1})
        assert resp.status_code == 200
        assert len(resp.json()["investigations"]) == 1
        default_resp = client.get("/v1/investigations")
        assert default_resp.status_code == 200
        assert len(default_resp.json()["investigations"]) >= 1
    finally:
        client.post(f"/v1/investigate/{inv_id}/cancel")


# ── Investigation feedback (P4-5c) ───────────────────────────────────────────


async def test_investigation_feedback_round_trip() -> None:
    from app.feedback import get_investigation_rating

    inv = await manager.create("feedback round trip query", "local")
    try:
        resp = client.post(f"/v1/investigate/{inv.id}/feedback", json={"rating": 4})
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"investigation_id": inv.id, "rating": 4, "stored": True}
        assert await get_investigation_rating(inv.id) == 4
    finally:
        await manager.cancel(inv.id)


async def test_investigation_feedback_rating_bounds() -> None:
    inv = await manager.create("feedback bounds query", "local")
    try:
        assert client.post(f"/v1/investigate/{inv.id}/feedback", json={"rating": 0}).status_code == 422
        assert client.post(f"/v1/investigate/{inv.id}/feedback", json={"rating": 6}).status_code == 422
    finally:
        await manager.cancel(inv.id)


def test_investigation_feedback_unknown_id_returns_404() -> None:
    resp = client.post("/v1/investigate/inv_nope_unknown/feedback", json={"rating": 5})
    assert resp.status_code == 404


async def test_investigation_feedback_503_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rediskit import holder

    inv = await manager.create("feedback no redis query", "local")
    try:
        monkeypatch.setattr(holder, "client", None)
        resp = client.post(f"/v1/investigate/{inv.id}/feedback", json={"rating": 3})
        assert resp.status_code == 503
    finally:
        monkeypatch.undo()
        await manager.cancel(inv.id)


async def test_investigation_feedback_metric_increments() -> None:
    from prometheus_client import REGISTRY

    inv = await manager.create("feedback metric query", "local")
    try:
        before = REGISTRY.get_sample_value(
            "argus_investigation_feedback_total", {"rating": "5"}
        ) or 0.0
        resp = client.post(f"/v1/investigate/{inv.id}/feedback", json={"rating": 5})
        assert resp.status_code == 200
        after = REGISTRY.get_sample_value(
            "argus_investigation_feedback_total", {"rating": "5"}
        ) or 0.0
        assert after == before + 1.0
    finally:
        await manager.cancel(inv.id)


# ── Supervisor sweep on restart (P4-5d) ───────────────────────────────────────


async def test_sweep_expires_overdue_row() -> None:
    from app.investigations import InvestigationManager

    fresh = InvestigationManager()
    inv = await fresh.create("sweep overdue query", "local")
    try:
        # Simulate a restart: drop in-memory supervision without touching the store.
        task = fresh._supervisors.pop(inv.id, None)
        if task is not None:
            task.cancel()
        fresh._events.pop(inv.id, None)
        inv.deadline_at = time.time() - 1.0
        await fresh._store.save(inv, settings.investigation_ttl_s)
        result = await fresh.sweep_expired()
        assert result["expired"] == 1
        assert result["rehydrated"] == 0
        loaded = await fresh.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert loaded.status_reason == StatusReason.WALL_CLOCK_LIMIT
    finally:
        await fresh.cancel(inv.id)


async def test_sweep_rehydrates_live_row_and_expiry_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.investigations import InvestigationManager

    monkeypatch.setattr(settings, "investigation_max_wall_time_s", 1)
    fresh = InvestigationManager()
    inv = await fresh.create("sweep live query", "local")
    try:
        task = fresh._supervisors.pop(inv.id, None)
        if task is not None:
            task.cancel()
        fresh._events.pop(inv.id, None)
        assert inv.id not in fresh._events
        assert inv.id not in fresh._supervisors
        result = await fresh.sweep_expired()
        assert result == {"expired": 0, "rehydrated": 1}
        assert inv.id in fresh._events
        assert inv.id in fresh._supervisors
        await asyncio.sleep(1.4)
        loaded = await fresh.get(inv.id)
        assert loaded is not None
        assert loaded.status == InvestigationStatus.BUDGET_EXHAUSTED
        assert loaded.status_reason == StatusReason.WALL_CLOCK_LIMIT
    finally:
        await fresh.cancel(inv.id)


async def test_sweep_is_idempotent() -> None:
    from app.investigations import InvestigationManager

    fresh = InvestigationManager()
    inv = await fresh.create("sweep idempotent query", "local")
    try:
        task = fresh._supervisors.pop(inv.id, None)
        if task is not None:
            task.cancel()
        fresh._events.pop(inv.id, None)
        first = await fresh.sweep_expired()
        assert first["expired"] == 0
        assert first["rehydrated"] == 1
        second = await fresh.sweep_expired()
        assert second == {"expired": 0, "rehydrated": 0}
    finally:
        await fresh.cancel(inv.id)


async def test_sweep_empty_store_returns_zeros() -> None:
    from app.evidence.store import EvidenceBoardStore
    from app.investigations import InvestigationManager

    fresh = InvestigationManager(EvidenceBoardStore())
    assert await fresh.sweep_expired() == {"expired": 0, "rehydrated": 0}
