import pytest
from fakeredis import aioredis as fakeredis_aioredis

from app.orchestration.report_jobs import ReportJobStore
from app.rediskit import holder


@pytest.fixture
async def fake_redis():
    client = fakeredis_aioredis.FakeRedis(decode_responses=True)
    holder.client = client
    yield client
    holder.client = None
    await client.aclose()


async def test_create_get_update_lifecycle():
    store = ReportJobStore()
    job = await store.create("Research topic X")
    assert job.status == "queued"

    updated = await store.update(job.job_id, status="running", progress="Planning")
    assert updated is not None and updated.status == "running"
    assert updated.updated_at >= job.created_at

    fetched = await store.get(job.job_id)
    assert fetched is not None and fetched.progress == "Planning"


async def test_update_unknown_job_returns_none():
    store = ReportJobStore()
    assert await store.update("missing", status="failed") is None


async def test_redis_mirror_survives_memory_loss(fake_redis):
    store = ReportJobStore()
    job = await store.create("Mirror me")
    await store.update(job.job_id, status="completed", result_markdown="# Done")

    fresh_store = ReportJobStore()
    restored = await fresh_store.get(job.job_id)
    assert restored is not None
    assert restored.status == "completed"
    assert restored.result_markdown == "# Done"


async def test_restore_corrupt_payload_returns_none(fake_redis):
    store = ReportJobStore()
    job = await store.create("x")
    await fake_redis.set(f"argus:report:{job.job_id}", "{corrupt")

    fresh_store = ReportJobStore()
    assert await fresh_store.get(job.job_id) is None


async def test_get_unknown_without_redis_returns_none():
    holder.client = None
    store = ReportJobStore()
    assert await store.get("nope") is None
