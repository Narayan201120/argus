from typing import Literal

from fastapi import APIRouter

from app.api.schemas import ConnectorHealthStatus, HealthResponse
from app.config import settings
from app.connectors.registry import registry
from app.rediskit import ping_redis
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

OverallStatus = Literal["ok", "degraded", "unavailable"]


async def _redis_status() -> Literal["ok", "unavailable", "disabled"]:
    if not settings.redis_enabled:
        return "disabled"
    return "ok" if await ping_redis() else "unavailable"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    statuses: list[ConnectorHealthStatus] = []
    redis_status = await _redis_status()

    for connector in registry.all():
        is_ok = await connector.health_check()
        connector.is_available = is_ok  # Update live availability flag
        statuses.append(ConnectorHealthStatus(
            connector_id=connector.connector_id,
            is_available=is_ok,
            status="ok" if is_ok else "unavailable",
        ))

    overall: OverallStatus
    if not statuses:
        overall = "unavailable"
    elif all(s.is_available for s in statuses):
        overall = "ok" if redis_status != "unavailable" else "degraded"
    elif any(s.is_available for s in statuses):
        overall = "degraded"
    else:
        overall = "unavailable"

    logger.info({"message": "Health check", "overall": overall, "redis": redis_status})
    return HealthResponse(status=overall, connectors=statuses, redis=redis_status)
