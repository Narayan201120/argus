from fastapi import APIRouter

from app.api.schemas import ModelsResponse, ConnectorProfile
from app.connectors.registry import registry

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    profiles = [
        ConnectorProfile(
            connector_id=c.connector_id,
            display_name=c.display_name,
            capabilities=c.capabilities,
            is_available=c.is_available,
        )
        for c in registry.all()
    ]
    return ModelsResponse(connectors=profiles, total=len(profiles))
