"""Stage P3-5 - A/B routing split + quality feedback (mock-only)."""

from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.config import settings
from app.connectors.base import BaseConnector
from app.main import app
from app.orchestration.binding import parse_router_split, pick_strategy_by_hash
from tests.test_memory import _attach_fake_redis as attach_fake_redis
from tests.test_memory import _use_connector

client = TestClient(app)


class UnusedProviderConnector(BaseConnector):
    connector_id = "unused"
    display_name = "Unused"
    capabilities = ["text"]
    is_available = True

    async def query(self, prompt, sub_query, config):  # pragma: no cover
        raise NotImplementedError

    async def health_check(self):
        return True


def test_parse_split_valid_and_normalizes():
    entries = parse_router_split("Semantic:80, static:20")
    assert entries == [("semantic", 80.0), ("static", 20.0)]


def test_parse_split_drops_unknown_and_invalid():
    assert parse_router_split("bogus:50,static:50") == [("static", 50.0)]
    assert parse_router_split("semantic:abc, static:x, semantic:10") == [("semantic", 10.0)]
    assert parse_router_split("   ") is None


def test_pick_strategy_is_deterministic():
    entries = parse_router_split("semantic:50,static:50")
    picks = {pick_strategy_by_hash(entries, f"query {i}") for i in range(200)}
    picks_again = {pick_strategy_by_hash(entries, f"query {i}") for i in range(200)}
    assert picks == picks_again
    assert picks <= {"semantic", "static"}


def test_ab_split_routes_all_traffic_to_full_weight_strategy(monkeypatch):
    attach_fake_redis(monkeypatch)
    stub = _use_connector(monkeypatch)
    monkeypatch.setattr(settings, "router_ab_split", "semantic:100")

    seen_strategies = []
    for text in ("question one about things", "question two about other stuff"):
        response = client.post("/v1/query", json={"query": text})
        assert response.status_code == 200
        seen_strategies.append(response.json()["router_strategy"])

    assert all(strategy == "semantic" for strategy in seen_strategies)

    value = REGISTRY.get_sample_value(
        "argus_router_decisions_total",
        {"method": "none", "matched_profile": "none"},
    )
    assert value is not None and value >= 2.0
    assert stub.seen_prompts  # pipeline actually ran


def test_explicit_strategy_beats_ab_split(monkeypatch):
    attach_fake_redis(monkeypatch)
    _use_connector(monkeypatch)
    monkeypatch.setattr(settings, "router_ab_split", "semantic:100")

    response = client.post("/v1/query", json={
        "query": "hello there friend",
        "model_config": {"router_strategy": "static"},
    })

    assert response.status_code == 200
    assert response.json()["router_strategy"] == "static"


def test_feedback_roundtrip_and_metrics(monkeypatch):
    attach_fake_redis(monkeypatch)

    request_id = "req-12345678"
    post = client.post("/v1/feedback", json={"request_id": request_id, "rating": 4})
    assert post.status_code == 200
    assert post.json() == {"request_id": request_id, "rating": 4, "stored": True}

    body = client.get("/v1/metrics").content
    assert b'argus_feedback_total{rating="4"}' in body

    fetched = client.get(f"/v1/feedback/{request_id}")
    assert fetched.status_code == 200
    assert fetched.json()["rating"] == 4


def test_feedback_rejects_out_of_range_rating():
    response = client.post("/v1/feedback", json={"request_id": "req-12345678", "rating": 9})
    assert response.status_code == 422


def test_feedback_get_returns_404_when_absent(monkeypatch):
    attach_fake_redis(monkeypatch)
    response = client.get("/v1/feedback/no-such-request-id-0001")
    assert response.status_code == 404


def test_feedback_unavailable_without_redis(monkeypatch):
    from app.rediskit import holder as redis_holder

    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    del fr
    monkeypatch.setattr(redis_holder, "client", None)
    monkeypatch.setattr(settings, "memory_enabled", True)

    response = client.post("/v1/feedback", json={"request_id": "req-12345678", "rating": 3})
    assert response.status_code == 503
