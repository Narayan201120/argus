"""Rate limiting middleware backed by Redis.

Two algorithms, selectable via RATE_LIMIT_ALGORITHM:
- fixed (default): INCR counter per window bucket - cheap, but allows a
  2x burst across window boundaries.
- sliding: exact sorted-set of hit timestamps per identity - smooth
  limits with no boundary bursts, slightly more Redis work.

Both are keyed per authenticated subject when available, else by client
IP. When Redis is unavailable the middleware fails open so a cache
outage never becomes an API outage.
"""

import time
import uuid

from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.metrics import RATE_LIMIT_REJECTIONS

EXEMPT_PATHS = frozenset({"/v1/health", "/v1/metrics"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, holder):
        super().__init__(app)
        self._holder = holder

    async def dispatch(self, request: Request, call_next) -> Response:
        client = self._holder.client
        if (
            client is None
            or not settings.rate_limit_enabled
            or request.url.path in EXEMPT_PATHS
        ):
            return await call_next(request)

        identity = identity_for(request)
        try:
            if settings.rate_limit_algorithm.lower() == "sliding":
                allowed, retry_after = await sliding_check(client, identity)
            else:
                allowed, retry_after = await fixed_check(client, identity)
        except RedisError:
            return await call_next(request)

        if not allowed:
            RATE_LIMIT_REJECTIONS.inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


def identity_for(request: Request) -> str:
    """Authenticated requests get a per-subject bucket so users behind a
    shared IP are not lumped together; anonymous requests fall back to IP."""
    subject = getattr(request.state, "subject", None)
    if subject:
        return f"sub:{subject}"
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


async def fixed_check(client, identity: str) -> tuple[bool, int]:
    """Classic INCR/EXPIRE window bucket. Returns (allowed, retry_after_s)."""
    window = max(settings.rate_limit_window_s, 1)
    window_start = int(time.time()) // window * window
    key = f"argus:rl:{identity}:{window_start}"

    current = await client.incr(key)
    if current == 1:
        await client.expire(key, window)

    if current > settings.rate_limit_max_requests:
        return False, max(window_start + window - int(time.time()), 1)
    return True, 0


async def sliding_check(client, identity: str, now: float | None = None) -> tuple[bool, int]:
    """Exact sliding window via a sorted set of hit timestamps.

    Returns (allowed, retry_after_s). Entries expire with the window, and
    the oldest remaining timestamp tells us exactly how long a rejected
    caller must wait.
    """
    now = time.time() if now is None else now
    window = max(settings.rate_limit_window_s, 1)
    key = f"argus:rlz:{identity}:{window}s"

    # Drop hits older than the window, count what remains.
    await client.zremrangebyscore(key, "-inf", now - window)
    current = await client.zcard(key)

    if current >= settings.rate_limit_max_requests:
        oldest = await client.zrange(key, 0, 0, withscores=True)
        oldest_score = oldest[0][1] if oldest else now
        retry_after = max(int(window - (now - oldest_score)) + 1, 1)
        return False, retry_after

    member = f"{now}:{uuid.uuid4().hex[:8]}"
    await client.zadd(key, {member: now})
    await client.expire(key, window)
    return True, 0


def retry_seconds() -> int:
    window = max(settings.rate_limit_window_s, 1)
    window_start = int(time.time()) // window * window
    return max(window_start + window - int(time.time()), 1)
