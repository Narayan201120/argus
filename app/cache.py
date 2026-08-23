"""Response caching for successful /v1/query results.

Cache keys hash the full effective request (query + model_config) so any
configuration change produces a distinct entry. All operations fail open:
cache errors never break request handling.
"""

import hashlib
import json
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import settings


class ResponseCache:
    def __init__(self, client: Redis):
        self._client = client

    @staticmethod
    def _key(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"argus:cache:{digest}"

    async def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            raw = await self._client.get(self._key(payload))
        except RedisError as exc:
            from app.utils.logger import get_logger

            get_logger(__name__).warning({"message": "Cache read failed", "error": str(exc)})
            return None
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, payload: dict[str, Any], response_body: dict[str, Any]) -> bool:
        if len(json.dumps(response_body).encode("utf-8")) > settings.cache_max_bytes:
            return False
        try:
            await self._client.set(
                self._key(payload),
                json.dumps(response_body),
                ex=settings.cache_ttl_s,
            )
            return True
        except RedisError as exc:
            from app.utils.logger import get_logger

            get_logger(__name__).warning({"message": "Cache write failed", "error": str(exc)})
            return False
