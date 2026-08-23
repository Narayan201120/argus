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
