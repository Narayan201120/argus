"""SSE streaming variant of the query pipeline.

Emits `role_complete` events as each parallel worker finishes, streamed
synthesis tokens (`synthesis_start` / `synthesis_token` / `synthesis_end`),
and a terminal `final` event carrying the same envelope as POST /v1/query.
Streaming responses are not cached.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from app.api.routes.query import _build_status, _token_usage_out
from app.api.routes.shared import resolve_request_connectors
from app.api.schemas import ModelStatus, QueryRequest, QueryResponse
from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorResponse
from app.metrics import record_role_outcome, record_role_tokens
from app.orchestration.aggregator import synthesize_stream
from app.orchestration.binding import binding_service
from app.orchestration.decomposer import _is_simple_query, build_parallel_plan
from app.orchestration.workers import (
    _query_with_retry,
    run_analysis_task,
    run_research_task,
    run_verification_task,
)
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

StreamRole = Literal["researcher", "analyzer", "verifier", "direct"]
ROLE_ORDER: tuple[StreamRole, ...] = ("researcher", "analyzer", "verifier")
TASK_ATTRS = {
    "researcher": "research_task",
    "analyzer": "analysis_task",
    "verifier": "verification_task",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _role_complete(
    role: StreamRole,
    connector_id: str,
    status: str,
    latency_ms: int,
    error: str | None,
    sub_query: str | None,
) -> dict:
    return {
        "role": role,
        "connector_id": connector_id,
        "status": status,
        "latency_ms": max(latency_ms, 0),
        "error": error,
        "sub_query": sub_query,
    }


def _model_status(role: StreamRole, response: ConnectorResponse) -> ModelStatus:
    return ModelStatus(
        role=role,
        connector_id=response.model_id,
        status=response.status.value,
        latency_ms=max(response.latency_ms, 0),
        error=response.error,
        token_usage=_token_usage_out(response.token_usage),
        sub_query=response.sub_query,
        retry_after_s=response.retry_after_s,
    )


@router.post("/query/stream")
async def stream_query(request: QueryRequest) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    resolved = await resolve_request_connectors(request)
    active_connectors = resolved.active_connectors
    overrides = resolved.overrides
    connector_config = ConnectorConfig(
        timeout_s=request.model_config_.timeout_s,
        max_tokens=request.model_config_.max_tokens,
        temperature=request.model_config_.temperature,
    )

    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    async def emit(event: str, data: dict) -> None:
        await queue.put((event, data))

    async def tracked(role: StreamRole, connector: BaseConnector, run_coro):
        """Run one worker; emit role_complete the moment it settles."""
        start = time.monotonic()
        try:
            outcome = await run_coro()
        except Exception as exc:
            await emit("role_complete", _role_complete(
                role=role,
                connector_id=connector.connector_id,
                status="error",
                latency_ms=int((time.monotonic() - start) * 1000),
                error=str(exc),
                sub_query=None,
            ))
            raise
        await emit("role_complete", _role_complete(
            role=role,
            connector_id=connector.connector_id,
            status="success",
            latency_ms=outcome.response.latency_ms,
            error=None,
            sub_query=outcome.response.sub_query,
        ))
        return outcome

    async def run_long_pipeline(total_start: float) -> QueryResponse:
        plan = build_parallel_plan(query=request.query, request_id=request_id)

        research_conn = binding_service.select_connector(active_connectors, "researcher", overrides=overrides)
        analysis_conn = binding_service.select_connector(active_connectors, "analyzer", overrides=overrides)
        verifier_conn = binding_service.select_connector(
            active_connectors,
            "verifier",
            {research_conn.connector_id, analysis_conn.connector_id},
            overrides=overrides,
        )
        conns = {
            "researcher": research_conn,
            "analyzer": analysis_conn,
            "verifier": verifier_conn,
        }
        tasks_by_role = {
            "researcher": lambda: run_research_task(
                research_conn, plan.shared_state, plan.research_task, connector_config,
            ),
            "analyzer": lambda: run_analysis_task(
                analysis_conn, plan.shared_state, plan.analysis_task, connector_config,
            ),
            "verifier": lambda: run_verification_task(
                verifier_conn, plan.shared_state, plan.verification_task, connector_config,
            ),
        }

        dispatch_start = time.monotonic()
        outputs = await asyncio.gather(
            *(tracked(role, conns[role], tasks_by_role[role]) for role in ROLE_ORDER),
            return_exceptions=True,
        )
        dispatch_ms = int((time.monotonic() - dispatch_start) * 1000)

        outputs_by_role = dict(zip(ROLE_ORDER, outputs, strict=True))
        bundle: dict[str, ConnectorResponse] = {}
        for role in ROLE_ORDER:
            task_obj = getattr(plan, TASK_ATTRS[role])
            bundle[role] = _build_status(
                connector_id=conns[role].connector_id,
                objective=task_obj.objective,
                output=outputs_by_role[role],
                fallback_latency_ms=dispatch_ms,
                role=role,
            )

        available_by_id = {c.connector_id: c for c in active_connectors}
        synth_chain = [
            available_by_id[cid]
            for cid in binding_service.preference_chain("synthesizer", overrides)
            if cid in available_by_id
        ]

        synthesizer_used = "fallback_concat"
        result_parts: list[str] = []
        async for kind, payload in synthesize_stream(
            original_query=request.query,
            response_bundle=bundle,
            synthesizer_chain=synth_chain,
            config=connector_config,
        ):
            if kind == "start":
                await emit("synthesis_start", {"connector_id": payload})
            elif kind == "token":
                result_parts.append(payload)
                await emit("synthesis_token", {"delta": payload})
            elif kind == "end":
                synthesizer_used = payload
                await emit("synthesis_end", {"connector_id": payload})
            else:  # fallback_concat
                synthesizer_used = "fallback_concat"
                result_parts = [payload]
                await emit("synthesis_fallback_concat", {})

        model_statuses = [_model_status(role, bundle[role]) for role in ROLE_ORDER]
        assignments: dict[str, str] = {role: conns[role].connector_id for role in ROLE_ORDER}
        assignments["synthesizer"] = synthesizer_used

        return QueryResponse(
            request_id=request_id,
            query=request.query,
            result="".join(result_parts),
            synthesizer=synthesizer_used,
            model_statuses=model_statuses,
            latency_breakdown={"total_ms": int((time.monotonic() - total_start) * 1000)},
            short_circuited=False,
            role_assignments=assignments,
            router_strategy=resolved.router_strategy,
            matched_profile=resolved.matched_profile,
        )

    async def run_short_pipeline(total_start: float) -> QueryResponse:
        direct = binding_service.select_connector(
            active_connectors, "direct", overrides=overrides,
        )
        response = await _query_with_retry(
            direct,
            prompt="You are the direct response layer of an AI orchestration system.",
            sub_query=request.query,
            config=connector_config,
        )
        record_role_outcome(
            "direct", response.model_id, response.status.value, response.latency_ms
        )
        record_role_tokens("direct", response.model_id, response.token_usage)
        await emit("role_complete", _role_complete(
            role="direct",
            connector_id=direct.connector_id,
            status=response.status.value,
            latency_ms=response.latency_ms,
            error=response.error,
            sub_query=response.sub_query,
        ))
        return QueryResponse(
            request_id=request_id,
            query=request.query,
            result=response.content or "Unable to generate a response.",
            synthesizer=direct.connector_id,
            model_statuses=[_model_status("direct", response)],
            latency_breakdown={"total_ms": int((time.monotonic() - total_start) * 1000)},
            short_circuited=True,
            role_assignments={"direct": direct.connector_id},
            router_strategy=resolved.router_strategy,
            matched_profile=resolved.matched_profile,
        )

    async def producer() -> None:
        total_start = time.monotonic()
        try:
            if _is_simple_query(request.query):
                envelope = await run_short_pipeline(total_start)
            else:
                envelope = await run_long_pipeline(total_start)
            await emit("final", envelope.model_dump(mode="json"))
        except Exception as exc:
            logger.error({
                "message": "Stream pipeline failed",
                "request_id": request_id,
                "error": str(exc),
            })
            await emit("stream_error", {"request_id": request_id, "error": str(exc)})
        finally:
            await queue.put(None)

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.get_running_loop().create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield _sse(event, data)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
