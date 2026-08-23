"""Fixed-window rate limiting middleware backed by Redis.

Keyed by client IP. When Redis is unavailable the middleware fails open so
a cache outage never becomes an API outage.
"""

import time

from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, holder):
        super().__init__(app)
        self._holder = holder

    async def dispatch(self, request: Request, call_next) -> Response:
        client = self._holder.client
        if (
            client is None
            or not settings.rate_limit_enabled
            or request.url.path == "/v1/health"
        ):
            return await call_next(request)

        try:
            current = await client.incr(key_for(request))
            if current == 1:
                window = max(settings.rate_limit_window_s, 1)
                await client.expire(key_for(request), window)
        except RedisError:
            return await call_next(request)

        if current > settings.rate_limit_max_requests:
            retry_after = retry_seconds()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


def key_for(request: Request) -> str:
    client_ip = request.client.host if request.client else "unknown"
    window = max(settings.rate_limit_window_s, 1)
    window_start = int(time.time()) // window * window
    return f"argus:rl:{client_ip}:{window_start}"


def retry_seconds() -> int:
    window = max(settings.rate_limit_window_s, 1)
    window_start = int(time.time()) // window * window
    return max(window_start + window - int(time.time()), 1)
