from fastapi import APIRouter

from app.api.schemas import (
    ConnectorProfile,
    ModelsResponse,
    RoutingInfoResponse,
    RoutingProfileOut,
    RoutingStrategyOut,
)
from app.connectors.availability import consecutive_auth_failures, is_demoted
from app.connectors.registry import registry
from app.orchestration.binding import ROUTER_STRATEGIES, binding_service

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    profiles = [
        ConnectorProfile(
            connector_id=c.connector_id,
            display_name=c.display_name,
            capabilities=c.capabilities,
            is_available=c.is_available,
            demoted=is_demoted(c.connector_id),
            consecutive_auth_failures=consecutive_auth_failures(c.connector_id),
        )
        for c in registry.all()
    ]
    return ModelsResponse(connectors=profiles, total=len(profiles))


@router.get("/routing", response_model=RoutingInfoResponse)
async def get_routing() -> RoutingInfoResponse:
    """Expose router strategies and named profiles for UI dropdowns."""
    strategies = [
        RoutingStrategyOut(name=name, description=description)
        for name, description in ROUTER_STRATEGIES.items()
    ]
    profiles = [
        RoutingProfileOut(
            name=name,
            connectors=list(definition.connectors),
            description=definition.description,
        )
        for name, definition in binding_service.config.profiles.items()
    ]
    return RoutingInfoResponse(strategies=strategies, profiles=profiles)
