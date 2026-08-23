import pytest
from fakeredis import aioredis as fakeredis_aioredis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.ratelimit import RateLimitMiddleware
from app.rediskit import holder


@pytest.fixture
async def fake_redis():
    client = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def _app() -> FastAPI:
    application = FastAPI()

    @application.get("/v1/health")
    async def health():
        return {"ok": True}

    @application.get("/ping")
    async def ping():
        return {"ok": True}

    return application


def test_rate_limit_blocks_after_max_requests(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_max_requests", 3)
    monkeypatch.setattr(settings, "rate_limit_window_s", 60)
    monkeypatch.setattr(holder, "client", fake_redis)

    client = TestClient(RateLimitMiddleware(_app(), holder))

    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    blocked = client.get("/ping")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1


def test_health_path_is_exempt(fake_redis, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_max_requests", 1)
    monkeypatch.setattr(settings, "rate_limit_window_s", 60)
    monkeypatch.setattr(holder, "client", fake_redis)

    client = TestClient(RateLimitMiddleware(_app(), holder))
    for _ in range(5):
        assert client.get("/v1/health").status_code == 200


def test_fail_open_without_redis(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_max_requests", 1)
    monkeypatch.setattr(holder, "client", None)

    client = TestClient(RateLimitMiddleware(_app(), holder))
    for _ in range(5):
        assert client.get("/ping").status_code == 200


def test_subject_keyed_buckets_separate_from_ip(fake_redis, monkeypatch):
    from starlette.requests import Request

    from app.ratelimit import identity_for

    monkeypatch.setattr(holder, "client", fake_redis)

    request = Request({"type": "http", "client": ("203.0.113.9", 1234)})
    ip_identity = identity_for(request)

    request.state.subject = "alice"
    alice_identity = identity_for(request)

    request.state.subject = "bob"
    bob_identity = identity_for(request)

    assert ip_identity == "ip:203.0.113.9"
    assert alice_identity == "sub:alice"
    assert bob_identity == "sub:bob"


@pytest.mark.asyncio
async def test_sliding_allows_max_then_blocks_and_recovers(fake_redis, monkeypatch):
    from app.ratelimit import sliding_check

    monkeypatch.setattr(settings, "rate_limit_max_requests", 3)

    base = 1000.0
    results = []
    for offset in range(3):  # three hits inside the window are allowed
        results.append(await sliding_check(fake_redis, "alice", now=base + offset))
    assert all(allowed for allowed, _ in results)

    allowed, retry_after = await sliding_check(fake_redis, "alice", now=base + 3)
    assert allowed is False
    assert retry_after >= 1

    # Once every hit ages out of the window, capacity returns.
    allowed_late, _ = await sliding_check(fake_redis, "alice", now=base + 61)
    assert allowed_late is True


@pytest.mark.asyncio
async def test_sliding_retry_after_derived_from_oldest_hit(fake_redis, monkeypatch):
    from app.ratelimit import sliding_check

    monkeypatch.setattr(settings, "rate_limit_max_requests", 2)

    base = 2000.0
    await sliding_check(fake_redis, "bob", now=base)          # expires at 2060
    allowed, _ = await sliding_check(fake_redis, "bob", now=base + 10)
    assert allowed is True

    allowed, retry_after = await sliding_check(fake_redis, "bob", now=base + 11)
    assert allowed is False
    # oldest hit at base -> free again at base+window; ceil-ish remainder:
    assert retry_after == int(60 - 11) + 1


@pytest.mark.asyncio
async def test_sliding_identities_are_independent(fake_redis, monkeypatch):
    from app.ratelimit import sliding_check

    monkeypatch.setattr(settings, "rate_limit_max_requests", 1)
    alice_allowed, _ = await sliding_check(fake_redis, "sub:alice", now=1000)
    bob_allowed, _ = await sliding_check(fake_redis, "sub:bob", now=1000)
    assert alice_allowed is True
    assert bob_allowed is True
