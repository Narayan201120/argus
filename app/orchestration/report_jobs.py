"""Job store for deep-report runs.

In-memory dict is the source of truth; when Redis is available each job
snapshot is mirrored so job status survives single-process restarts and is
readable from other workers. All Redis failures degrade to memory-only.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field, replace

from redis.exceptions import RedisError

from app.rediskit import holder
from app.utils.logger import get_logger

logger = get_logger(__name__)

JOB_TTL_S = 86400  # 24h


@dataclass
class ReportJob:
    job_id: str
    query: str
    status: str = "queued"  # queued | running | completed | failed
    progress: str = "Queued"
    result_markdown: str | None = None
    error: str | None = None
    role_assignments: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ReportJobStore:
    def __init__(self):
        self._jobs: dict[str, ReportJob] = {}

    async def create(self, query: str) -> ReportJob:
        job = ReportJob(job_id=str(uuid.uuid4()), query=query)
        self._jobs[job.job_id] = job
        await self._mirror(job)
        return job

    async def get(self, job_id: str) -> ReportJob | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        return await self._restore(job_id)

    async def update(self, job_id: str, **changes) -> ReportJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        changes.pop("job_id", None)
        updated = replace(job, updated_at=time.time(), **changes)
        self._jobs[job_id] = updated
        await self._mirror(updated)
        return updated

    def _key(self, job_id: str) -> str:
        return f"argus:report:{job_id}"

    async def _mirror(self, job: ReportJob) -> None:
        client = holder.client
        if client is None:
            return
        try:
            await client.set(
                self._key(job.job_id),
                json.dumps(asdict(job)),
                ex=JOB_TTL_S,
            )
        except RedisError as exc:
            logger.warning({"message": "Report job mirror failed", "error": str(exc)})

    async def _restore(self, job_id: str) -> ReportJob | None:
        client = holder.client
        if client is None:
            return None
        try:
            raw = await client.get(self._key(job_id))
        except RedisError as exc:
            logger.warning({"message": "Report job restore failed", "error": str(exc)})
            return None
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            known = {f for f in ReportJob.__dataclass_fields__}
            job = ReportJob(**{k: v for k, v in data.items() if k in known})
            self._jobs[job.job_id] = job
            return job
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning({"message": "Corrupt report job payload", "error": str(exc)})
            return None


report_job_store = ReportJobStore()
