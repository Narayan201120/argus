import asyncio
import time
import uuid
from typing import Literal

from fastapi import APIRouter

from app.api.routes.shared import resolve_request_connectors
from app.api.schemas import ModelStatus, QueryRequest, QueryResponse, TokenUsageOut
from app.cache import ResponseCache
from app.config import settings
from app.connectors.base import (
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.memory import load_history_text, session_store
from app.metrics import (
    CACHE_OPERATIONS,
    record_role_outcome,
    record_role_tokens,
)
from app.orchestration.aggregator import synthesize
from app.orchestration.binding import binding_service
from app.orchestration.decomposer import _is_simple_query, build_parallel_plan
from app.orchestration.workers import (
    RoleTaskError,
    WorkerOutcome,
    run_analysis_task,
    run_connector_query,
    run_research_task,
    run_verification_task,
)
from app.rediskit import holder
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

WorkerRole = Literal["researcher", "analyzer", "verifier"]
ROLE_ORDER: tuple[WorkerRole, ...] = ("researcher", "analyzer", "verifier")


def _build_status(
    connector_id: str,
    objective: str,
    output: WorkerOutcome | BaseException,
    fallback_latency_ms: int,
    role: str,
) -> ConnectorResponse:
    if isinstance(output, RoleTaskError):
        # Preserve the provider's TRUE status (rate_limited/timeout/...)
        # plus its retry hint instead of collapsing to generic error.
        source = output.response
        response = ConnectorResponse(
            model_id=connector_id,
            content="",
            latency_ms=max(source.latency_ms, 0),
            token_usage=source.token_usage or TokenUsage(),
            status=source.status,
            error=source.error,
            sub_query=objective,
            retry_after_s=source.retry_after_s,
        )
    elif isinstance(output, BaseException):
        response = ConnectorResponse(
            model_id=connector_id,
            content="",
            latency_ms=fallback_latency_ms,
            token_usage=TokenUsage(),
            status=ConnectorStatus.ERROR,
            error=str(output),
            sub_query=objective,
        )
    else:
        response = ConnectorResponse(
            model_id=connector_id,
            content=output.result.model_dump_json(indent=2),
            latency_ms=max(output.response.latency_ms, 0),
            token_usage=output.response.token_usage,
            status=ConnectorStatus.SUCCESS,
            sub_query=objective,
        )

    record_role_outcome(role, response.model_id, response.status.value, response.latency_ms)
    record_role_tokens(role, response.model_id, response.token_usage)
    return response


def _token_usage_out(token_usage: TokenUsage | None) -> TokenUsageOut | None:
    if token_usage is None:
        return None
    return TokenUsageOut(
        prompt_tokens=token_usage.prompt_tokens,
        completion_tokens=token_usage.completion_tokens,
        total_tokens=token_usage.total_tokens,
    )


def _cache_payload(request: QueryRequest) -> dict:
    return {
        "query": request.query,
        "model_config": request.model_config_.model_dump(exclude_none=True),
    }


def _response_cache() -> ResponseCache | None:
    if settings.cache_enabled and holder.available:
        return holder.cache
    return None


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    request_id = str(uuid.uuid4())
    total_start = time.monotonic()

    cache = _response_cache()
    cache_payload = _cache_payload(request)
    if cache is not None:
        cached_body = await cache.get(cache_payload)
        if cached_body is not None:
            CACHE_OPERATIONS.labels(result="hit").inc()
            cached_response = QueryResponse.model_validate(cached_body)
            logger.info({
                "message": "Query served from cache",
                "request_id": request_id,
                "cached_request_id": cached_response.request_id,
            })
            cached_response.cache_hit = True
            return cached_response
        CACHE_OPERATIONS.labels(result="miss").inc()

    mc = request.model_config_
    resolved = await resolve_request_connectors(request)
    active_connectors = resolved.active_connectors
    role_binding_overrides = resolved.overrides

    logger.info({
        "message": "Query received",
        "request_id": request_id,
        "query_length": len(request.query),
        "requested_connectors": [c.connector_id for c in active_connectors],
        "profile": mc.profile,
        "router_strategy": resolved.router_strategy,
        "matched_profile": resolved.matched_profile,
        "role_bindings": role_binding_overrides or None,
    })

    connector_config = ConnectorConfig(
        timeout_s=mc.timeout_s,
        max_tokens=mc.max_tokens,
        temperature=mc.temperature,
    )
    available_by_id = {connector.connector_id: connector for connector in active_connectors}
    synthesizer_chain = [
        available_by_id[connector_id]
        for connector_id in binding_service.preference_chain("synthesizer", role_binding_overrides)
        if connector_id in available_by_id
    ]
    role_assignments: dict[str, str] = {}

    session_id = request.session_id
    history_text = await load_history_text(session_id)

    decompose_start = time.monotonic()
    short_circuited = _is_simple_query(request.query)
    decompose_ms = int((time.monotonic() - decompose_start) * 1000)

    if short_circuited:
        # Failover chain: preference order restricted to the active pool.
        direct_chain = [
            available_by_id[connector_id]
            for connector_id in binding_service.preference_chain("direct", role_binding_overrides)
            if connector_id in available_by_id
        ] or list(active_connectors)
        if not settings.direct_failover:
            direct_chain = direct_chain[:1]

        direct_response: ConnectorResponse | None = None
        direct_connector = direct_chain[0]
        # Working memory: short follow-ups benefit from history too.
        direct_sub_query = (
            f"{history_text}\n\nCurrent question: {request.query}"
            if history_text
            else request.query
        )
        for candidate in direct_chain:
            direct_connector = candidate
            role_assignments["direct"] = direct_connector.connector_id
            candidate_response = await run_connector_query(
                direct_connector,
                prompt="You are the direct response layer of an AI orchestration system.",
                sub_query=direct_sub_query,
                config=connector_config,
            )
            record_role_outcome(
                "direct",
                candidate_response.model_id,
                candidate_response.status.value,
                candidate_response.latency_ms,
            )
            record_role_tokens("direct", candidate_response.model_id, candidate_response.token_usage)
            direct_response = candidate_response
            if candidate_response.status == ConnectorStatus.SUCCESS:
                break
            # Non-success (rate_limited/timeout/error): try the next
            # provider in the chain instead of failing the request.

        assert direct_response is not None  # chain is non-empty by construction
        total_ms = int((time.monotonic() - total_start) * 1000)
        response = QueryResponse(
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
                    retry_after_s=direct_response.retry_after_s,
                )
            ],
            latency_breakdown={"decompose_ms": decompose_ms, "total_ms": total_ms},
            short_circuited=True,
            role_assignments=role_assignments,
            router_strategy=resolved.router_strategy,
            matched_profile=resolved.matched_profile,
            session_id=session_id,
        )
        if direct_response.status == ConnectorStatus.SUCCESS:
            await session_store.append(session_id or "", request.query, response.result)
        if cache is not None and direct_response.status == ConnectorStatus.SUCCESS:
            await cache.set(cache_payload, response.model_dump(mode="json"))
        return response

    plan = build_parallel_plan(
        query=request.query,
        request_id=request_id,
        conversation_history=history_text,
    )

    dispatch_start = time.monotonic()
    research_connector = binding_service.select_connector(
        active_connectors, "researcher", overrides=role_binding_overrides,
    )
    analysis_connector = binding_service.select_connector(
        active_connectors, "analyzer", overrides=role_binding_overrides,
    )
    verification_connector = binding_service.select_connector(
        active_connectors,
        "verifier",
        {research_connector.connector_id, analysis_connector.connector_id},
        overrides=role_binding_overrides,
    )
    role_assignments.update({
        "researcher": research_connector.connector_id,
        "analyzer": analysis_connector.connector_id,
        "verifier": verification_connector.connector_id,
    })

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

    response_bundle: dict[str, ConnectorResponse] = {
        "researcher": _build_status(
            connector_id=research_connector.connector_id,
            objective=plan.research_task.objective,
            output=research_output,
            fallback_latency_ms=dispatch_ms,
            role="researcher",
        ),
        "analyzer": _build_status(
            connector_id=analysis_connector.connector_id,
            objective=plan.analysis_task.objective,
            output=analysis_output,
            fallback_latency_ms=dispatch_ms,
            role="analyzer",
        ),
        "verifier": _build_status(
            connector_id=verification_connector.connector_id,
            objective=plan.verification_task.objective,
            output=verification_output,
            fallback_latency_ms=dispatch_ms,
            role="verifier",
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
            connector_id=response_bundle[role].model_id,
            status=response_bundle[role].status.value,
            latency_ms=response_bundle[role].latency_ms,
            error=response_bundle[role].error,
            token_usage=_token_usage_out(response_bundle[role].token_usage),
            sub_query=response_bundle[role].sub_query,
            retry_after_s=response_bundle[role].retry_after_s,
        )
        for role in ROLE_ORDER
    ]

    total_ms = int((time.monotonic() - total_start) * 1000)
    role_assignments["synthesizer"] = synthesizer_used
    logger.info({
        "message": "Query complete",
        "request_id": request_id,
        "synthesizer": synthesizer_used,
        "role_assignments": role_assignments,
        "total_ms": total_ms,
    })

    response = QueryResponse(
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
        role_assignments=role_assignments,
        router_strategy=resolved.router_strategy,
        matched_profile=resolved.matched_profile,
        session_id=session_id,
    )
    if result:
        await session_store.append(session_id or "", request.query, result)
    if cache is not None:
        await cache.set(cache_payload, response.model_dump(mode="json"))
    return response
