"""Stage P6-0 - feedback store decoupled from working-memory flag, mock-only."""

import pytest
from fakeredis import aioredis as fakeredis_aioredis

from app.config import settings
from app.feedback import (
    get_investigation_rating,
    get_rating,
    save_investigation_rating,
    save_rating,
)


@pytest.fixture
async def fake_redis():
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield fr
    await fr.aclose()


@pytest.fixture(autouse=True)
def _use_fake_redis(fake_redis, monkeypatch):
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", fake_redis)
    monkeypatch.setattr(settings, "feedback_enabled", True)


async def test_answer_rating_roundtrip():
    assert await save_rating("req-1", 4) is True
    assert await get_rating("req-1") == 4


async def test_investigation_rating_roundtrip():
    assert await save_investigation_rating("inv-1", 5) is True
    assert await get_investigation_rating("inv-1") == 5


async def test_ratings_work_with_memory_disabled(monkeypatch):
    monkeypatch.setattr(settings, "memory_enabled", False)
    assert await save_rating("req-2", 3) is True
    assert await get_rating("req-2") == 3
    assert await save_investigation_rating("inv-2", 2) is True
    assert await get_investigation_rating("inv-2") == 2


async def test_ratings_off_with_feedback_disabled(monkeypatch):
    monkeypatch.setattr(settings, "feedback_enabled", False)
    assert await save_rating("req-3", 5) is False
    assert await get_rating("req-3") is None
    assert await save_investigation_rating("inv-3", 5) is False
    assert await get_investigation_rating("inv-3") is None
