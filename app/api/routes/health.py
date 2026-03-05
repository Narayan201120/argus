from fastapi import APIRouter

from app.api.schemas import HealthResponse, ConnectorHealthStatus
from app.connectors.registry import registry
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    statuses = []

    for connector in registry.all():
        is_ok = await connector.health_check()
        connector.is_available = is_ok  # Update live availability flag
        statuses.append(ConnectorHealthStatus(
            connector_id=connector.connector_id,
            is_available=is_ok,
            status="ok" if is_ok else "unavailable",
        ))

    if not statuses:
        overall = "unavailable"
    elif all(s.is_available for s in statuses):
        overall = "ok"
    elif any(s.is_available for s in statuses):
        overall = "degraded"
    else:
        overall = "unavailable"

    logger.info({"message": "Health check", "overall": overall})
    return HealthResponse(status=overall, connectors=statuses)
