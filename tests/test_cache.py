import pytest
from fakeredis import aioredis as fakeredis_aioredis

from app.cache import ResponseCache


@pytest.fixture
async def fake_redis():
    client = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def cache(fake_redis):
    return ResponseCache(fake_redis)


async def test_cache_roundtrip(cache):
    payload = {"query": "hello world", "model_config": {"timeout_s": 45}}
    body = {"result": "the answer", "synthesizer": "mistral"}

    assert await cache.get(payload) is None
    assert await cache.set(payload, body) is True
    assert await cache.get(payload) == body


async def test_cache_key_is_payload_sensitive(cache):
    base = {"query": "q", "model_config": {"timeout_s": 45}}
    variant = {"query": "q", "model_config": {"timeout_s": 60}}

    await cache.set(base, {"result": "first"})
    await cache.set(variant, {"result": "second"})

    assert (await cache.get(base))["result"] == "first"
    assert (await cache.get(variant))["result"] == "second"


async def test_cache_skips_oversized_bodies(cache):
    payload = {"query": "big"}
    oversized = {"result": "x" * (262145)}
    assert await cache.set(payload, oversized) is False
    assert await cache.get(payload) is None


async def test_cache_fail_open_on_error(cache, fake_redis, monkeypatch):
    from redis.exceptions import RedisError

    payload = {"query": "bye"}
    await cache.set(payload, {"result": "kept"})

    async def boom(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(fake_redis, "get", boom)
    assert await cache.get(payload) is None
