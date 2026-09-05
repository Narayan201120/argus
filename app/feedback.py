"""Quality-feedback store for A/B routing experiments (Redis, fail-open).

One rating per request_id: ``argus:feedback:{request_id}`` -> JSON
``{"rating": n, "ts": epoch}``. Ratings pair with the strategy recorded
in each answer's envelope (argus_router_decisions_total / router_strategy)
so strategies can be compared on real usage.
"""

import json
import time
from typing import Any

from app.config import settings
from app.rediskit import holder
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _key(request_id: str) -> str:
    return f"argus:feedback:{request_id}"


def _inv_key(investigation_id: str) -> str:
    return f"argus:feedback:inv:{investigation_id}"


async def save_rating(request_id: str, rating: int) -> bool:
    """Store a rating. Returns True when persisted."""
    if not settings.memory_enabled or not request_id:
        return False
    client = holder.client
    if client is None:
        return False
    try:
        payload: dict[str, Any] = {"rating": rating, "ts": time.time()}
        ttl = max(settings.memory_ttl_s, 60)
        await client.set(_key(request_id), json.dumps(payload), ex=ttl)
        return True
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning({"message": "Feedback save failed (ignored)", "error": str(exc)})
        return False


async def get_rating(request_id: str) -> int | None:
    if not settings.memory_enabled or not request_id:
        return None
    client = holder.client
    if client is None:
        return None
    try:
        raw = await client.get(_key(request_id))
        return json.loads(raw)["rating"] if raw else None
    except Exception as exc:  # noqa: BLE001
        logger.warning({"message": "Feedback read failed (ignored)", "error": str(exc)})
        return None


async def save_investigation_rating(investigation_id: str, rating: int) -> bool:
    """Store an investigation report rating. Returns True when persisted."""
    if rating < 1 or rating > 5:
        raise ValueError("rating must be between 1 and 5")
    if not settings.memory_enabled or not investigation_id:
        return False
    client = holder.client
    if client is None:
        return False
    try:
        payload: dict[str, Any] = {"rating": rating, "ts": time.time()}
        ttl = max(settings.memory_ttl_s, 60)
        await client.set(_inv_key(investigation_id), json.dumps(payload), ex=ttl)
        return True
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning({"message": "Investigation feedback save failed (ignored)", "error": str(exc)})
        return False


async def get_investigation_rating(investigation_id: str) -> int | None:
    if not settings.memory_enabled or not investigation_id:
        return None
    client = holder.client
    if client is None:
        return None
    try:
        raw = await client.get(_inv_key(investigation_id))
        return json.loads(raw)["rating"] if raw else None
    except Exception as exc:  # noqa: BLE001
        logger.warning({"message": "Investigation feedback read failed (ignored)", "error": str(exc)})
        return None
