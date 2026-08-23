"""Role-to-provider binding.

Loads routing preferences from a YAML config so role bindings and named
profiles can be changed without code changes. Falls back to built-in
defaults when the file is missing or malformed, mirroring the prompt
loader fallback pattern.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import settings
from app.connectors.base import BaseConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)

RoutingRole = str

DEFAULT_ROLES: dict[str, list[str]] = {
    "researcher": ["gemini", "openai", "claude", "mistral"],
    "analyzer": ["openai", "mistral", "claude", "gemini"],
    "verifier": ["claude", "mistral", "openai", "gemini"],
    "synthesizer": ["claude", "openai", "gemini", "mistral"],
    "direct": ["claude", "openai", "gemini", "mistral"],
    "planner": ["openai", "claude", "gemini", "mistral"],
    "writer": ["claude", "openai", "gemini", "mistral"],
    "reviewer": ["claude", "openai", "mistral", "gemini"],
}

DEFAULT_PROFILES: dict[str, list[str]] = {
    "research": ["gemini", "claude", "mistral", "openai"],
    "code": ["openai", "claude", "mistral", "gemini"],
    "analysis": ["claude", "openai", "mistral", "gemini"],
    "fast": ["mistral", "gemini"],
}

# Router strategies selectable via settings.router_strategy or a
# per-request model_config.router_strategy override. 'static' preserves
# the exact pre-Stage-6 behavior; 'semantic' infers a named profile from
# the query text when the caller did not set one explicitly.
ROUTER_STRATEGIES: dict[str, str] = {
    "static": "Fixed YAML preference chains per role (Stage 1 behavior).",
    "semantic": "Keyword intent classifier picks a named profile per query.",
}

# Intent lexicon for the semantic router. Keys must name profiles that
# exist in RoutingConfig.profiles; unknown names are ignored at inference
# time so removing a profile in YAML also removes it from semantic reach.
PROFILE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "research": (
        "research", "literature", "survey", "sources", "evidence",
        "state of the art", "papers", "history of", "compare",
    ),
    "code": (
        "code", "bug", "stack trace", "refactor", "compile", "regex",
        "python", "typescript", "sql", "docker", "function", "api endpoint",
    ),
    "analysis": (
        "analyze", "tradeoff", "trade-offs", "risk", "strategy",
        "evaluate", "pros and cons", "decision", "benchmark", "optimize",
        "architecture",
    ),
    "fast": ("quick", "briefly", "short answer", "tl;dr", "one-liner"),
}


@dataclass(frozen=True)
class RoutingConfig:
    roles: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_ROLES.items()}
    )
    profiles: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PROFILES.items()}
    )


def load_routing_config(path: str | Path) -> RoutingConfig:
    path = Path(path)
    if not path.exists():
        logger.warning({"message": f"{path} not found, using default routing config"})
        return RoutingConfig()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        logger.warning({"message": f"Malformed routing config at {path}, using defaults", "error": str(exc)})
        return RoutingConfig()

    roles = {
        role: [str(connector_id) for connector_id in chain]
        for role, chain in (raw.get("roles") or {}).items()
        if isinstance(chain, list)
    }
    profiles = {
        profile: [str(connector_id) for connector_id in connectors]
        for profile, connectors in (raw.get("profiles") or {}).items()
        if isinstance(connectors, list)
    }

    merged_roles = {k: list(v) for k, v in DEFAULT_ROLES.items()}
    merged_roles.update(roles)
    merged_profiles = {k: list(v) for k, v in DEFAULT_PROFILES.items()}
    merged_profiles.update(profiles)

    return RoutingConfig(roles=merged_roles, profiles=merged_profiles)


class RoleBindingService:
    """Selects providers for orchestration roles from configured chains."""

    def __init__(self, config: RoutingConfig):
        self._config = config

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RoleBindingService":
        return cls(load_routing_config(path or settings.routing_config_path))

    @property
    def config(self) -> RoutingConfig:
        return self._config

    def preference_chain(self, role: str, overrides: dict[str, list[str]] | None = None) -> list[str]:
        if overrides and role in overrides:
            return [str(connector_id) for connector_id in overrides[role]]
        return self._config.roles.get(role, [])

    def known_profile(self, profile: str) -> bool:
        return profile in self._config.profiles

    def known_strategy(self, strategy: str) -> bool:
        return strategy in ROUTER_STRATEGIES

    def infer_profile(self, query: str | None) -> str | None:
        """Classify a query into a named routing profile.

        Scores each configured profile by counting case-insensitive
        keyword hits from PROFILE_KEYWORDS. Highest score wins; ties go
        to the earlier profile in PROFILE_KEYWORDS order. Returns None
        when no keyword matches or the query is empty.
        """
        if not query:
            return None

        normalized = query.lower()
        best_profile: str | None = None
        best_score = 0
        for profile, keywords in PROFILE_KEYWORDS.items():
            if profile not in self._config.profiles:
                continue
            score = sum(1 for keyword in keywords if keyword in normalized)
            if score > best_score:
                best_profile = profile
                best_score = score
        return best_profile

    def profile_connectors(self, profile: str) -> list[str]:
        return list(self._config.profiles.get(profile, []))

    def select_connector(
        self,
        active_connectors: list[BaseConnector],
        role: str,
        excluded_ids: set[str] | None = None,
        overrides: dict[str, list[str]] | None = None,
    ) -> BaseConnector:
        excluded_ids = excluded_ids or set()
        by_id = {connector.connector_id: connector for connector in active_connectors}

        for connector_id in self.preference_chain(role, overrides):
            connector = by_id.get(connector_id)
            if connector is not None and connector.connector_id not in excluded_ids:
                return connector

        for connector in active_connectors:
            if connector.connector_id not in excluded_ids:
                return connector

        # A single provider may legitimately fill every role once
        # exclusions apply; reuse it rather than failing the request.
        if active_connectors:
            return active_connectors[-1]

        raise ValueError(f"No available connector for role '{role}'")


binding_service = RoleBindingService.load()
