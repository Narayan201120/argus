"""Stage P3-3 - working memory (Redis sessions), mock-only."""

import json

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.memory import format_history, session_store

client = TestClient(app)


@pytest.fixture
async def fake_redis():
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield fr
    await fr.aclose()


@pytest.fixture(autouse=True)
def _use_fake_redis(fake_redis, monkeypatch):
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", fake_redis)
    monkeypatch.setattr(settings, "memory_enabled", True)


def _key(sid: str) -> str:
    return f"argus:sess:{sid}"


async def test_append_stores_and_trims_to_max_turns(fake_redis):
    for i in range(14):
        await session_store.append("s1", f"q{i}", f"a{i}")
    raw = json.loads(await fake_redis.get(_key("s1")))
    assert len(raw) == settings.memory_max_turns  # 12
    assert raw[-1]["q"] == "q13"
    assert raw[0]["q"] == "q2"  # oldest rolled off


async def test_recent_returns_inject_turns_newest_last(fake_redis):
    for i in range(6):
        await session_store.append("s2", f"q{i}", f"a{i}")
    recent = await session_store.recent("s2")  # default inject turns = 4
    assert [t["q"] for t in recent] == ["q2", "q3", "q4", "q5"]


async def test_ttl_is_refreshed_on_append(fake_redis):
    await session_store.append("ttl", "q", "a")
    ttl = await fake_redis.ttl(_key("ttl"))
    assert 0 < ttl <= settings.memory_ttl_s


async def test_clear_removes_session(fake_redis):
    await session_store.append("gone", "q", "a")
    assert await session_store.clear("gone") is True
    assert await session_store.recent("gone") == []


async def test_fail_open_when_redis_absent(monkeypatch):
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", None)
    await session_store.append("x", "q", "a")  # must not raise
    assert await session_store.recent("x") == []


def test_format_history_respects_char_cap(monkeypatch):
    monkeypatch.setattr(settings, "memory_char_cap", 200)
    turns = [
        {"q": f"question number {i} with padding text", "a": f"answer {i} also padded out", "ts": i}
        for i in range(5)
    ]
    formatted = format_history(turns)
    assert formatted is not None
    # newest kept preferentially; at least one oldest dropped
    assert "answer 4" in formatted
    assert "answer 0" not in formatted


# ── API integration ─────────────────────────────────────────────────────────


class StubConnector:
    connector_id = "stub"
    display_name = "Stub"
    capabilities = ["text"]
    is_available = True

    def __init__(self):
        self.seen_prompts: list[str] = []

    async def query(self, prompt, sub_query, config):
        self.seen_prompts.append(sub_query)
        from app.connectors.base import ConnectorResponse, ConnectorStatus, TokenUsage

        return ConnectorResponse(
            model_id="stub-model",
            content="Direct response alpha",
            latency_ms=1,
            token_usage=TokenUsage(1, 1, 2),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


def _attach_fake_redis(monkeypatch):
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    from app.rediskit import holder as redis_holder

    monkeypatch.setattr(redis_holder, "client", fr)
    return fr


def _use_connector(monkeypatch):
    stub = StubConnector()
    from app.connectors.registry import registry

    monkeypatch.setattr(registry, "_connectors", {"stub": stub})
    return stub


def test_second_query_receives_first_exchange_via_history(monkeypatch):
    _attach_fake_redis(monkeypatch)
    stub = _use_connector(monkeypatch)

    first = client.post("/v1/query", json={
        "query": "What is a LLM?",
        "session_id": "sess-abc",
    })
    assert first.status_code == 200
    assert first.json()["session_id"] == "sess-abc"

    second = client.post("/v1/query", json={
        "query": "explain that in more detail",
        "session_id": "sess-abc",
    })
    assert second.status_code == 200

    # Short follow-ups take the direct path, which must carry history.
    direct_prompt = next(p for p in stub.seen_prompts if "Earlier conversation" in p)
    assert "Direct response alpha" in direct_prompt
    assert "Current question: explain that in more detail" in direct_prompt


def test_different_session_ids_are_isolated(monkeypatch):
    _attach_fake_redis(monkeypatch)
    stub = _use_connector(monkeypatch)

    client.post("/v1/query", json={"query": "secret topic", "session_id": "sess-one"})
    second = client.post("/v1/query", json={"query": "hello there", "session_id": "sess-two"})
    assert second.status_code == 200

    # sess-two has no history -> plain sub-query, no injected transcript
    hello_direct = [p for p in stub.seen_prompts if p == "hello there"]
    assert hello_direct

    got_one = client.get("/v1/session/sess-one").json()
    got_two = client.get("/v1/session/sess-two").json()
    assert [t["q"] for t in got_one["turns"]] == ["secret topic"]
    assert [t["q"] for t in got_two["turns"]] == ["hello there"]  # own content only


def test_session_endpoints_roundtrip(fake_redis):
    import asyncio

    async def seed():
        await session_store.append("sess-view", "q1", "a1")

    asyncio.run(seed())
    got = client.get("/v1/session/sess-view")
    assert got.status_code == 200
    assert got.json()["turns"][0]["q"] == "q1"

    cleared = client.delete("/v1/session/sess-view")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True
    assert client.get("/v1/session/sess-view").json()["turns"] == []
