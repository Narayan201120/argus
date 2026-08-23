"""Shared async Redis access.

The client is created during app lifespan startup. A module-level holder
keeps the client reachable from middleware and services without import-time
side effects, and lets tests inject a fakeredis instance.
"""

from collections.abc import Awaitable

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.cache import ResponseCache
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisHolder:
    def __init__(self):
        self.client: Redis | None = None
        self.cache: ResponseCache | None = None

    @property
    def available(self) -> bool:
        return self.client is not None


holder = RedisHolder()


async def connect_redis() -> Redis | None:
    if not settings.redis_enabled:
        logger.info({"message": "Redis disabled by configuration"})
        return None
    try:
        from redis.asyncio import ConnectionPool

        pool = ConnectionPool.from_url(settings.redis_url, decode_responses=True)
        client = Redis(connection_pool=pool)
        if not await _drain(client.ping()):
            logger.warning({"message": "Redis PING failed", "url": settings.redis_url})
            return None
        logger.info({"message": "Redis connected", "url": settings.redis_url})
        return client
    except (RedisError, OSError) as exc:
        logger.warning({"message": "Redis unavailable, continuing without it", "error": str(exc)})
        return None


async def _drain(value: Awaitable[bool] | bool) -> bool:
    if isinstance(value, bool):
        return value
    return bool(await value)


async def close_redis() -> None:
    if holder.client is not None:
        try:
            await holder.client.aclose()
        except RedisError as exc:
            logger.warning({"message": "Redis close error", "error": str(exc)})
        holder.client = None


async def ping_redis() -> bool:
    """Ping the shared client, tolerating sync/async SDK return shapes."""
    if holder.client is None:
        return False
    try:
        return await _drain(holder.client.ping())
    except RedisError:
        return False
