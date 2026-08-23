import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.api.routes.shared import resolve_request_connectors
from app.api.schemas import (
    QueryRequest,
    ReportCreateResponse,
    ReportJobStatus,
)
from app.connectors.base import ConnectorConfig
from app.orchestration.report_jobs import report_job_store
from app.orchestration.report_runner import execute_report
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/report", status_code=202, response_model=ReportCreateResponse)
async def create_report(request: QueryRequest) -> ReportCreateResponse:
    resolved = resolve_request_connectors(request)
    active = resolved.active_connectors
    job = await report_job_store.create(request.query)

    config = ConnectorConfig(
        timeout_s=request.model_config_.timeout_s,
        max_tokens=request.model_config_.max_tokens,
        temperature=request.model_config_.temperature,
    )
    logger.info({
        "message": "Report job accepted",
        "job_id": job.job_id,
        "query_length": len(request.query),
        "connectors": [c.connector_id for c in active],
        "router_strategy": resolved.router_strategy,
        "matched_profile": resolved.matched_profile,
    })

    asyncio.create_task(
        execute_report(
            job_id=job.job_id,
            query=request.query,
            config=config,
            overrides=request.model_config_.role_bindings,
            active_connectors=active,
        )
    )
    return ReportCreateResponse(job_id=job.job_id, poll_url=f"/v1/report/{job.job_id}")


@router.get("/report/{job_id}", response_model=ReportJobStatus)
async def get_report(job_id: str) -> ReportJobStatus:
    job = await report_job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown report job.")
    return ReportJobStatus(**asdict(job))
