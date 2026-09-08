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

from redis.exceptions import RedisError, WatchError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.metrics import RATE_LIMIT_REJECTIONS

EXEMPT_PATHS = frozenset({"/v1/health", "/v1/metrics"})

# Optimistic-locking attempts for the sliding window. Each aborted attempt
# implies a rival committed a write, and commits stop once the cap is hit,
# so a small bound converges; exhausting it raises WatchError (a RedisError,
# so the middleware fails open per contract).
_SLIDING_MAX_ATTEMPTS = 10


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
    """Classic INCR/EXPIRE window bucket. Returns (allowed, retry_after_s).

    INCR and EXPIRE share one MULTI/EXEC pipeline so a crash or a race
    between the two can never leave a counter behind with no TTL. EXPIRE
    uses NX so concurrent hits never push the window forward: the bucket
    keeps the fixed boundary derived from ``window_start``.
    """
    window = max(settings.rate_limit_window_s, 1)
    window_start = int(time.time()) // window * window
    key = f"argus:rl:{identity}:{window_start}"

    pipe = client.pipeline(transaction=True)
    pipe.incr(key)
    pipe.expire(key, window, nx=True)
    current, _ = await pipe.execute()
    current = int(current)

    if current > settings.rate_limit_max_requests:
        return False, max(window_start + window - int(time.time()), 1)
    return True, 0


async def sliding_check(client, identity: str, now: float | None = None) -> tuple[bool, int]:
    """Exact sliding window via a sorted set of hit timestamps.

    Returns (allowed, retry_after_s). Entries expire with the window, and
    the oldest remaining timestamp tells us exactly how long a rejected
    caller must wait.

    The read-check-write sequence runs under WATCH/MULTI with bounded
    retries so concurrent hits cannot all observe a stale count and
    over-admit past the cap. (A Lua script would do this in one round
    trip, but the mock Redis used in tests has no EVAL support, and
    WATCH/MULTI is atomic on real Redis too.) Only reads happen between
    WATCH and MULTI; the trim plus the write ride inside the transaction,
    because Redis aborts EXEC when the watching client itself writes to a
    watched key first.
    """
    now = time.time() if now is None else now
    window = max(settings.rate_limit_window_s, 1)
    cap = settings.rate_limit_max_requests
    key = f"argus:rlz:{identity}:{window}s"
    cutoff = now - window
    member = f"{now}:{uuid.uuid4().hex[:8]}"

    for _ in range(_SLIDING_MAX_ATTEMPTS):
        pipe = client.pipeline(transaction=True)
        try:
            await pipe.watch(key)
            # Read-only: ZRANGE returns members ordered by score, so the
            # first in-window entry is the oldest, exactly as after the
            # old ZREMRANGEBYSCORE + ZRANGE 0 0 sequence (the trim bound
            # is inclusive, hence survivors are scores strictly > cutoff).
            entries = await pipe.zrange(key, 0, -1, withscores=True)
            fresh = [(m, float(score)) for m, score in entries if float(score) > cutoff]

            if len(fresh) >= cap:
                oldest_score = fresh[0][1] if fresh else now
                retry_after = max(int(window - (now - oldest_score)) + 1, 1)
                return False, retry_after

            pipe.multi()
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.zadd(key, {member: now})
            pipe.expire(key, window)
            await pipe.execute()
            return True, 0
        except WatchError:
            continue
        finally:
            await pipe.reset()
    raise WatchError("sliding rate-limit check lost too many races; failing open")


def retry_seconds() -> int:
    window = max(settings.rate_limit_window_s, 1)
    window_start = int(time.time()) // window * window
    return max(window_start + window - int(time.time()), 1)
