import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.api.schemas import ModelStatus, QueryRequest, QueryResponse, TokenUsageOut
from app.connectors.base import ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.orchestration.aggregator import synthesize
from app.orchestration.analyzer import run_analysis_task
from app.orchestration.decomposer import _is_simple_query, build_parallel_plan
from app.orchestration.researcher import run_research_task
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _select_research_connector(active_connectors):
    return (
        registry.get("gemini")
        or registry.get("openai")
        or active_connectors[0]
    )


def _select_analysis_connector(active_connectors):
    return (
        registry.get("openai")
        or registry.get("mistral")
        or registry.get("claude")
        or active_connectors[min(1, len(active_connectors) - 1)]
    )


def _build_status(
    role_id: str,
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


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    total_start = time.monotonic()

    logger.info({
        "message": "Query received",
        "request_id": request_id,
        "query_length": len(request.query),
        "requested_connectors": request.model_config_.connectors,
    })

    active_connectors = [
        registry.get(cid)
        for cid in request.model_config_.connectors
        if registry.get(cid) and registry.get(cid).is_available
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
        c for c in [
            registry.get("claude"),
            registry.get("openai"),
            registry.get("gemini"),
        ]
        if c is not None and c.is_available
    ]

    decompose_start = time.monotonic()
    short_circuited = _is_simple_query(request.query)
    decompose_ms = int((time.monotonic() - decompose_start) * 1000)

    if short_circuited:
        logger.info({"message": "Short-circuit: aggregator answering directly", "request_id": request_id})
        result, synthesizer_used, _ = await synthesize(
            original_query=request.query,
            response_bundle={},
            synthesizer_chain=synthesizer_chain,
            config=connector_config,
        )
        return QueryResponse(
            request_id=request_id,
            query=request.query,
            result=result or "Unable to generate a response.",
            synthesizer=synthesizer_used,
            model_statuses=[],
            latency_breakdown={"total_ms": int((time.monotonic() - total_start) * 1000)},
            short_circuited=True,
        )

    plan = build_parallel_plan(
        query=request.query,
        request_id=request_id,
    )

    dispatch_start = time.monotonic()
    research_connector = _select_research_connector(active_connectors)
    analysis_connector = _select_analysis_connector(active_connectors)

    research_output, analysis_output = await asyncio.gather(
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
        return_exceptions=True,
    )
    dispatch_ms = int((time.monotonic() - dispatch_start) * 1000)

    response_bundle = {
        "researcher": _build_status(
            role_id="researcher",
            connector_id=research_connector.connector_id,
            objective=plan.research_task.objective,
            output=research_output,
            latency_ms=dispatch_ms,
        ),
        "analyzer": _build_status(
            role_id="analyzer",
            connector_id=analysis_connector.connector_id,
            objective=plan.analysis_task.objective,
            output=analysis_output,
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
            connector_id=cid,
            status=r.status.value,
            latency_ms=r.latency_ms,
            error=r.error,
            token_usage=TokenUsageOut(
                prompt_tokens=r.token_usage.prompt_tokens,
                completion_tokens=r.token_usage.completion_tokens,
                total_tokens=r.token_usage.total_tokens,
            ) if r.token_usage else None,
            sub_query=r.sub_query,
        )
        for cid, r in response_bundle.items()
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
