"""Shared request-resolution helpers for API routes."""

from dataclasses import dataclass

from fastapi import HTTPException

from app.api.schemas import QueryRequest
from app.config import settings
from app.connectors.base import BaseConnector
from app.connectors.registry import registry
from app.metrics import ROUTER_DECISIONS
from app.orchestration.binding import binding_service, semantic_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

OVERRIDABLE_ROLES = frozenset({"researcher", "analyzer", "verifier", "synthesizer"})


@dataclass(frozen=True)
class ResolvedRouting:
    """Outcome of request-level routing resolution, shared by all routes."""

    active_connectors: list[BaseConnector]
    overrides: dict[str, list[str]]
    router_strategy: str
    matched_profile: str | None


async def resolve_request_connectors(request: QueryRequest) -> ResolvedRouting:
    """Validate model_config and resolve the active connector list.

    Applies the selected router strategy ('static' keeps fixed YAML
    chains; 'semantic' infers a named profile from the query via
    embeddings with keyword fallback when the caller set none). Raises
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

    router_strategy = mc.router_strategy or settings.router_strategy
    if not binding_service.known_strategy(router_strategy):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown router strategy: {router_strategy}",
        )

    matched_profile: str | None = None
    effective_profile = mc.profile
    if router_strategy == "semantic" and effective_profile is None:
        matched_profile, method = await semantic_router.infer_profile(request.query)
        ROUTER_DECISIONS.labels(
            method=method, matched_profile=matched_profile or "none"
        ).inc()
        if matched_profile is not None:
            effective_profile = matched_profile
            logger.info({
                "message": "Semantic router inferred profile",
                "profile": matched_profile,
                "method": method,
                "query_length": len(request.query),
            })

    if mc.connectors:
        requested_ids = list(mc.connectors)
    elif effective_profile:
        if not binding_service.known_profile(effective_profile):
            raise HTTPException(status_code=422, detail=f"Unknown profile: {effective_profile}")
        requested_ids = binding_service.profile_connectors(effective_profile)
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

    return ResolvedRouting(
        active_connectors=active,
        overrides=overrides,
        router_strategy=router_strategy,
        matched_profile=matched_profile,
    )
