import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.api.schemas import ModelStatus, QueryRequest, QueryResponse, TokenUsageOut
from app.connectors.base import ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.orchestration.aggregator import synthesize
from app.orchestration.decomposer import _is_simple_query, build_parallel_plan
from app.orchestration.workers import run_analysis_task, run_research_task, run_verification_task
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

ROLE_PREFERENCES = {
    "researcher": ["gemini", "openai", "claude", "mistral"],
    "analyzer": ["openai", "mistral", "claude", "gemini"],
    "verifier": ["claude", "mistral", "openai", "gemini"],
    "direct": ["claude", "openai", "gemini", "mistral"],
}


def _select_connector(active_connectors, role: str, excluded_ids: set[str] | None = None):
    excluded_ids = excluded_ids or set()
    by_id = {connector.connector_id: connector for connector in active_connectors}

    for connector_id in ROLE_PREFERENCES[role]:
        connector = by_id.get(connector_id)
        if connector is not None and connector.connector_id not in excluded_ids:
            return connector

    for connector in active_connectors:
        if connector.connector_id not in excluded_ids:
            return connector

    return active_connectors[-1]


def _build_status(
    connector_id: str,
    objective: str,
    output,
    latency_ms: int,
) -> ConnectorResponse:
    if isinstance(output, Exception):
        return ConnectorResponse(
            model_id=connector_id,
            content="",
            latency_ms=latency_ms,
            token_usage=TokenUsage(),
            status=ConnectorStatus.ERROR,
            error=str(output),
            sub_query=objective,
        )

    return ConnectorResponse(
        model_id=connector_id,
        content=output.model_dump_json(indent=2),
        latency_ms=latency_ms,
        token_usage=TokenUsage(),
        status=ConnectorStatus.SUCCESS,
        sub_query=objective,
    )


def _token_usage_out(token_usage: TokenUsage | None) -> TokenUsageOut | None:
    if token_usage is None:
        return None
    return TokenUsageOut(
        prompt_tokens=token_usage.prompt_tokens,
        completion_tokens=token_usage.completion_tokens,
        total_tokens=token_usage.total_tokens,
    )


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    total_start = time.monotonic()
    requested_connector_ids = request.model_config_.connectors or registry.ids()

    logger.info({
        "message": "Query received",
        "request_id": request_id,
        "query_length": len(request.query),
        "requested_connectors": requested_connector_ids,
    })

    unknown_connector_ids = sorted(set(requested_connector_ids) - set(registry.ids()))
    if unknown_connector_ids:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown connector IDs: {', '.join(unknown_connector_ids)}",
        )

    active_connectors = [
        registry.get(connector_id)
        for connector_id in requested_connector_ids
        if registry.get(connector_id) and registry.get(connector_id).is_available
    ]

    if not active_connectors:
        raise HTTPException(
            status_code=503,
            detail="No available connectors for the requested model_config.",
        )

    connector_config = ConnectorConfig(
        timeout_s=request.model_config_.timeout_s,
        max_tokens=request.model_config_.max_tokens,
        temperature=request.model_config_.temperature,
    )
    synthesizer_chain = [
        connector
        for connector_id in ROLE_PREFERENCES["direct"]
        for connector in active_connectors
        if connector.connector_id == connector_id
    ]

    decompose_start = time.monotonic()
    short_circuited = _is_simple_query(request.query)
    decompose_ms = int((time.monotonic() - decompose_start) * 1000)

    if short_circuited:
        direct_connector = _select_connector(active_connectors, "direct")
        direct_response = await direct_connector.query(
            prompt="You are the direct response layer of an AI orchestration system.",
            sub_query=request.query,
            config=connector_config,
        )
        total_ms = int((time.monotonic() - total_start) * 1000)
        return QueryResponse(
            request_id=request_id,
            query=request.query,
            result=direct_response.content or "Unable to generate a response.",
            synthesizer=direct_connector.connector_id,
            model_statuses=[
                ModelStatus(
                    role="direct",
                    connector_id=direct_connector.connector_id,
                    status=direct_response.status.value,
                    latency_ms=direct_response.latency_ms,
                    error=direct_response.error,
                    token_usage=_token_usage_out(direct_response.token_usage),
                    sub_query=request.query,
                )
            ],
            latency_breakdown={"decompose_ms": decompose_ms, "total_ms": total_ms},
            short_circuited=True,
        )

    plan = build_parallel_plan(query=request.query, request_id=request_id)

    dispatch_start = time.monotonic()
    research_connector = _select_connector(active_connectors, "researcher")
    analysis_connector = _select_connector(active_connectors, "analyzer")
    verification_connector = _select_connector(
        active_connectors,
        "verifier",
        {research_connector.connector_id, analysis_connector.connector_id},
    )

    research_output, analysis_output, verification_output = await asyncio.gather(
        run_research_task(
            connector=research_connector,
            shared_state=plan.shared_state,
            task=plan.research_task,
            config=connector_config,
        ),
        run_analysis_task(
            connector=analysis_connector,
            shared_state=plan.shared_state,
            task=plan.analysis_task,
            config=connector_config,
        ),
        run_verification_task(
            connector=verification_connector,
            shared_state=plan.shared_state,
            task=plan.verification_task,
            config=connector_config,
        ),
        return_exceptions=True,
    )
    dispatch_ms = int((time.monotonic() - dispatch_start) * 1000)

    response_bundle = {
        "researcher": _build_status(
            connector_id=research_connector.connector_id,
            objective=plan.research_task.objective,
            output=research_output,
            latency_ms=dispatch_ms,
        ),
        "analyzer": _build_status(
            connector_id=analysis_connector.connector_id,
            objective=plan.analysis_task.objective,
            output=analysis_output,
            latency_ms=dispatch_ms,
        ),
        "verifier": _build_status(
            connector_id=verification_connector.connector_id,
            objective=plan.verification_task.objective,
            output=verification_output,
            latency_ms=dispatch_ms,
        ),
    }

    synthesis_start = time.monotonic()
    result, synthesizer_used, _ = await synthesize(
        original_query=request.query,
        response_bundle=response_bundle,
        synthesizer_chain=synthesizer_chain,
        config=connector_config,
    )
    synthesis_ms = int((time.monotonic() - synthesis_start) * 1000)

    model_statuses = [
        ModelStatus(
            role=role,
            connector_id=response.model_id,
            status=response.status.value,
            latency_ms=response.latency_ms,
            error=response.error,
            token_usage=_token_usage_out(response.token_usage),
            sub_query=response.sub_query,
        )
        for role, response in response_bundle.items()
    ]

    total_ms = int((time.monotonic() - total_start) * 1000)
    logger.info({
        "message": "Query complete",
        "request_id": request_id,
        "synthesizer": synthesizer_used,
        "total_ms": total_ms,
    })

    return QueryResponse(
        request_id=request_id,
        query=request.query,
        result=result,
        synthesizer=synthesizer_used,
        model_statuses=model_statuses,
        latency_breakdown={
            "decompose_ms": decompose_ms,
            "dispatch_ms": dispatch_ms,
            "synthesis_ms": synthesis_ms,
            "total_ms": total_ms,
        },
    )
