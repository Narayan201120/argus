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
        # Read the route template only after call_next: Starlette matches
        # the route inside the downstream router, so scope["route"] does
        # not exist yet when dispatch() is entered.
        route_template = getattr(request.scope.get("route"), "path", "unmatched")
        HTTP_REQUESTS.labels(method=method, path=route_template, status=status).inc()
        HTTP_LATENCY.labels(method=method, path=route_template).observe(duration)


@router.get("/metrics")
async def metrics() -> Response:
    """Expose the Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
