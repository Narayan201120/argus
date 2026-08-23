"""Shared request-resolution helpers for API routes."""

from fastapi import HTTPException

from app.api.schemas import QueryRequest
from app.connectors.base import BaseConnector
from app.connectors.registry import registry
from app.orchestration.binding import binding_service

OVERRIDABLE_ROLES = frozenset({"researcher", "analyzer", "verifier", "synthesizer"})


def resolve_request_connectors(
    request: QueryRequest,
) -> tuple[list[BaseConnector], dict[str, list[str]]]:
    """Validate model_config and resolve the active connector list.

    Returns (active_connectors, role_binding_overrides). Raises
    HTTPException(422/503) with the canonical error messages shared by
    the query, stream, and report routes.
    """
    mc = request.model_config_
    overrides = mc.role_bindings or {}

    invalid_roles = sorted(set(overrides) - OVERRIDABLE_ROLES)
    if invalid_roles:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown roles in role_bindings: {', '.join(invalid_roles)}",
        )

    if mc.connectors:
        requested_ids = list(mc.connectors)
    elif mc.profile:
        if not binding_service.known_profile(mc.profile):
            raise HTTPException(status_code=422, detail=f"Unknown profile: {mc.profile}")
        requested_ids = binding_service.profile_connectors(mc.profile)
    else:
        requested_ids = registry.ids()

    unknown_ids = sorted(set(requested_ids) - set(registry.ids()))
    if unknown_ids:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown connector IDs: {', '.join(unknown_ids)}",
        )

    active: list[BaseConnector] = []
    for connector_id in requested_ids:
        connector = registry.get(connector_id)
        if connector is not None and connector.is_available:
            active.append(connector)

    if not active:
        raise HTTPException(
            status_code=503,
            detail="No available connectors for the requested model_config.",
        )

    return active, overrides
