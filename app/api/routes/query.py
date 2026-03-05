import uuid
import time

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, ModelStatus, TokenUsageOut
from app.connectors.base import ConnectorConfig, ConnectorStatus
from app.connectors.registry import registry
from app.orchestration.decomposer import decompose_query
from app.orchestration.dispatcher import dispatch
from app.orchestration.aggregator import synthesize
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


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

    # ── Resolve active connectors ────────────────────────────────────────────
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

    # ── Synthesizer fallback chain ───────────────────────────────────────────
    synthesizer_chain = [
        c for c in [
            registry.get("claude"),
            registry.get("openai"),
            registry.get("gemini"),
        ]
        if c is not None and c.is_available
    ]

    # ── Step 1: Decompose ────────────────────────────────────────────────────
    decompose_start = time.monotonic()

    # Prefer openai for decomposition (fast + good structured output), else fallback
    decomposer = (
        registry.get("openai")
        or registry.get("gemini")
        or active_connectors[0]
    )

    sub_queries = await decompose_query(
        query=request.query,
        connectors=active_connectors,
        decomposer_connector=decomposer,
    )
    decompose_ms = int((time.monotonic() - decompose_start) * 1000)
    short_circuited = sub_queries is None

    # ── Step 2: Short-circuit path ───────────────────────────────────────────
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

    # ── Step 3: Dispatch ─────────────────────────────────────────────────────
    dispatch_start = time.monotonic()
    connector_map = {c.connector_id: c for c in active_connectors}

    response_bundle = await dispatch(
        prompt=request.query,
        sub_queries=sub_queries,
        connectors=connector_map,
        config=connector_config,
    )
    dispatch_ms = int((time.monotonic() - dispatch_start) * 1000)

    # ── Step 4: Synthesize ───────────────────────────────────────────────────
    synthesis_start = time.monotonic()
    result, synthesizer_used, _ = await synthesize(
        original_query=request.query,
        response_bundle=response_bundle,
        synthesizer_chain=synthesizer_chain,
        config=connector_config,
    )
    synthesis_ms = int((time.monotonic() - synthesis_start) * 1000)

    # ── Build response ───────────────────────────────────────────────────────
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
