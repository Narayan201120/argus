"""Prometheus metrics registry and HTTP instrumentation.

All metric objects live here as module-level singletons so tests and the
/v1/metrics scrape endpoint observe the same process-wide registry.
Cardinality guard: HTTP labels use the matched route template
("unmatched" for 404s), never raw request paths.
"""

import time

from fastapi import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.connectors.base import TokenUsage

router = APIRouter()

HTTP_REQUESTS = Counter(
    "argus_http_requests_total",
    "HTTP requests by method, route template, and status code.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "argus_http_request_duration_seconds",
    "End-to-end request duration in seconds.",
    ["method", "path"],
)
IN_FLIGHT = Gauge(
    "argus_http_in_flight",
    "Requests currently being served.",
)
CACHE_OPERATIONS = Counter(
    "argus_cache_operations_total",
    "Response cache lookups by result.",
    ["result"],
)
RATE_LIMIT_REJECTIONS = Counter(
    "argus_rate_limit_rejections_total",
    "Requests rejected with 429 by the rate limiter.",
)
ROLE_OUTCOMES = Counter(
    "argus_role_outcomes_total",
    "Worker role executions by outcome.",
    ["role", "connector_id", "status"],
)
ROLE_LATENCY = Histogram(
    "argus_role_latency_seconds",
    "Per-role connector-reported latency in seconds.",
    ["role", "connector_id"],
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
ROLE_TOKENS = Counter(
    "argus_role_tokens_total",
    "Provider-reported tokens consumed per role.",
    ["role", "connector_id", "type"],
)
REPORT_JOBS = Counter(
    "argus_report_jobs_total",
    "Deep-report jobs by terminal state.",
    ["status"],
)
ROUTER_DECISIONS = Counter(
    "argus_router_decisions_total",
    "Semantic routing decisions by deciding mechanism and matched profile.",
    ["method", "matched_profile"],
)
TRANSCRIPTIONS = Counter(
    "argus_transcriptions_total",
    "Audio transcription attempts by outcome.",
    ["status"],
)
TRANSCRIPTION_LATENCY = Histogram(
    "argus_transcription_latency_seconds",
    "End-to-end transcription duration in seconds.",
)
SPEECH_TOTAL = Counter(
    "argus_speech_total",
    "Text-to-speech attempts by outcome.",
    ["status"],
)
SPEECH_LATENCY = Histogram(
    "argus_speech_latency_seconds",
    "End-to-end speech synthesis duration in seconds.",
)
MEMORY_TRUNCATED_ANSWERS = Counter(
    "argus_memory_truncated_answers_total",
    "Answers stored shorter than generated because they exceeded the memory per-answer cap.",
)
FEEDBACK_TOTAL = Counter(
    "argus_feedback_total",
    "Quality feedback ratings received, by rating value.",
    ["rating"],
)
INVESTIGATIONS_TOTAL = Counter(
    "argus_investigations_total",
    "Investigation lifecycle events by event type.",
    ["event"],
)
TOOL_CALLS = Counter(
    "argus_tool_calls_total",
    "Investigation tool calls by tool name and outcome.",
    ["tool", "status"],
)
TOOL_LATENCY = Histogram(
    "argus_tool_latency_seconds",
    "Tool call latency in seconds by tool name.",
    ["tool"],
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
FIRST_EVIDENCE_LATENCY = Histogram(
    "argus_time_to_first_evidence_seconds",
    "Seconds from investigation creation to first evidence write.",
    buckets=(1, 2.5, 5, 10, 30, 60, 120),
)
WORKER_CALLS = Counter(
    "argus_worker_calls_total",
    "Analysis worker executions by worker name and outcome.",
    ["worker", "status"],
)
WORKER_LATENCY = Histogram(
    "argus_worker_latency_seconds",
    "Analysis worker latency in seconds by worker name.",
    ["worker"],
    buckets=(1, 2.5, 5, 10, 30, 60, 120, 300),
)
LOOP_STOPS = Counter(
    "argus_loop_stops_total",
    "Investigation loop endings by stop reason.",
    ["reason"],
)


def record_role_outcome(role: str, connector_id: str, status: str, latency_ms: int) -> None:
    """Record one worker-role execution outcome."""
    ROLE_OUTCOMES.labels(role=role, connector_id=connector_id, status=status).inc()
    ROLE_LATENCY.labels(role=role, connector_id=connector_id).observe(max(latency_ms, 0) / 1000)


def record_role_tokens(
    role: str, connector_id: str, token_usage: TokenUsage | None
) -> None:
    """Accumulate provider-reported token usage for a role, when present."""
    if token_usage is None or token_usage.total_tokens <= 0:
        return
    ROLE_TOKENS.labels(role=role, connector_id=connector_id, type="prompt").inc(
        token_usage.prompt_tokens
    )
    ROLE_TOKENS.labels(role=role, connector_id=connector_id, type="completion").inc(
        token_usage.completion_tokens
    )


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Outermost observability layer: counts and times every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method
        IN_FLIGHT.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Routing may not have populated scope["route"]; fall back.
            self._record(request, method, "500", time.perf_counter() - start)
            raise
        finally:
            IN_FLIGHT.dec()

        self._record(request, method, str(response.status_code), time.perf_counter() - start)
        return response

    @staticmethod
    def _record(request: Request, method: str, status: str, duration: float) -> None:
        HTTP_REQUESTS.labels(
            method=method, path=route_template_for(request), status=status
        ).inc()
        HTTP_LATENCY.labels(method=method, path=route_template_for(request)).observe(duration)


def route_template_for(request: Request) -> str:
    """Resolve the cardinality-safe route template for a request.

    Starlette >=1.x keeps include_router prefixes inside private wrapper
    objects, so scope["route"].path can be missing the mount prefix
    (e.g. "/query" instead of "/v1/query"). We therefore re-resolve the
    full path through each top-level route entry's url_path_for() using
    the matched route's name and path params, which works across
    versions. Falls back to scope["route"].path, then "unmatched".
    """
    route = request.scope.get("route")
    if route is None:
        return "unmatched"

    name = getattr(route, "name", None)
    params = request.scope.get("path_params") or {}
    if name:
        app = request.scope.get("app")
        for entry in getattr(app, "routes", []) or []:
            url_path_for = getattr(entry, "url_path_for", None)
            if url_path_for is None:
                continue
            try:
                return str(url_path_for(name, **params))
            except Exception:  # noqa: BLE001 - any non-match just means "try next"
                continue

    return getattr(route, "path", "unmatched")


@router.get("/metrics")
async def metrics() -> Response:
    """Expose the Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
