"""Role-to-provider binding.

Loads routing preferences from a YAML config so role bindings and named
profiles can be changed without code changes. Falls back to built-in
defaults when the file is missing or malformed, mirroring the prompt
loader fallback pattern.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import settings
from app.connectors.base import BaseConnector
from app.embeddings import cosine_similarity, get_embedder
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

# Router strategies selectable via settings.router_strategy or a
# per-request model_config.router_strategy override. 'static' preserves
# the exact pre-Stage-6 behavior; 'semantic' infers a named profile from
# the query text when the caller did not set one explicitly.
ROUTER_STRATEGIES: dict[str, str] = {
    "static": "Fixed YAML preference chains per role (Stage 1 behavior).",
    "semantic": "Intent classifier picks a named profile per query.",
}


def parse_router_split(raw: str | None) -> list[tuple[str, float]] | None:
    """Parse an A/B split string like "semantic:80,static:20".

    Returns cumulative-weight entries for known strategies only, or None
    when the setting is empty/unusable (fail-open to the default
    strategy).
    """
    if not raw or not raw.strip():
        return None
    entries: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, _, weight_raw = part.partition(":")
        name = name.strip().lower()
        try:
            weight = float(weight_raw)
        except ValueError:
            continue
        if name in ROUTER_STRATEGIES and weight > 0:
            entries.append((name, weight))
    return entries or None


def pick_strategy_by_hash(
    entries: list[tuple[str, float]], query_text: str
) -> str:
    """Deterministically assign a strategy by hashing the query text.

    The same question always lands in the same group, which keeps
    A/B comparisons reproducible.
    """
    total = sum(weight for _, weight in entries) or 1.0
    digest = hashlib.sha256(query_text.encode("utf-8")).digest()
    point = (int.from_bytes(digest[:8], "big") / 2**64) * total
    cumulative = 0.0
    for name, weight in entries:
        cumulative += weight
        if point < cumulative:
            return name
    return entries[-1][0]

# Intent lexicon fallback for built-in profiles. Profiles may override it
# per-profile via routing.yaml (`keywords:`); custom profiles without a
# lexicon entry are never inferred by the keyword classifier.
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

_DEFAULT_PROFILE_CONNECTORS: dict[str, list[str]] = {
    "research": ["gemini", "claude", "mistral", "openai"],
    "code": ["openai", "claude", "mistral", "gemini"],
    "analysis": ["claude", "openai", "mistral", "gemini"],
    "fast": ["mistral", "gemini"],
}


@dataclass(frozen=True)
class ProfileDefinition:
    """A named routing profile: provider pool plus optional intent hints."""

    connectors: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    description: str = ""


def _default_profiles() -> dict[str, ProfileDefinition]:
    return {
        name: ProfileDefinition(
            connectors=tuple(connectors),
            keywords=PROFILE_KEYWORDS.get(name, ()),
        )
        for name, connectors in _DEFAULT_PROFILE_CONNECTORS.items()
    }


@dataclass(frozen=True)
class RoutingConfig:
    roles: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_ROLES.items()}
    )
    profiles: dict[str, ProfileDefinition] = field(default_factory=_default_profiles)


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

    profiles: dict[str, ProfileDefinition] = {}
    for name, entry in (raw.get("profiles") or {}).items():
        # List form:  profile: [connector, ...]
        # Rich form:  profile: {connectors: [...], keywords: [...]}
        if isinstance(entry, list):
            profiles[str(name)] = ProfileDefinition(connectors=tuple(str(c) for c in entry))
        elif isinstance(entry, dict) and isinstance(entry.get("connectors"), list):
            keywords = entry.get("keywords")
            raw_description = entry.get("description")
            profiles[str(name)] = ProfileDefinition(
                connectors=tuple(str(c) for c in entry["connectors"]),
                keywords=(
                    tuple(str(k) for k in keywords) if isinstance(keywords, list) else ()
                ),
                description=str(raw_description) if raw_description else "",
            )

    merged_roles = {k: list(v) for k, v in DEFAULT_ROLES.items()}
    merged_roles.update(roles)
    merged_profiles = _default_profiles()
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
        for profile_name, definition in self._config.profiles.items():
            # Per-profile YAML lexicon wins; built-in fallback otherwise.
            lexicon = definition.keywords or PROFILE_KEYWORDS.get(profile_name, ())
            score = sum(1 for keyword in lexicon if keyword in normalized)
            if score > best_score:
                best_profile = profile_name
                best_score = score
        return best_profile

    def profile_connectors(self, profile: str) -> list[str]:
        definition = self._config.profiles.get(profile)
        return list(definition.connectors) if definition else []

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

SEMANTIC_EMBEDDING_COOLDOWN_S = 60.0


class SemanticRouter:
    """Embeddings-first intent classifier with keyword fallback.

    On each 'semantic' lookup: embed the query once, cosine-match against
    cached profile-description vectors, and accept the best match when it
    clears ROUTER_EMBEDDING_THRESHOLD. Any failure (no key, provider
    error, quota) starts a cooldown during which only the keyword
    classifier runs - a broken embedding backend can never fail requests.
    """

    def __init__(
        self,
        service: RoleBindingService,
        embedder_factory=None,
        cooldown_s: float = SEMANTIC_EMBEDDING_COOLDOWN_S,
    ):
        self._service = service
        self._embedder_factory = embedder_factory or get_embedder
        self._cooldown_s = cooldown_s
        self._profile_vectors: dict[str, list[float]] | None = None
        self._cooldown_until = 0.0

    @staticmethod
    def _description_for(name: str, definition) -> str:
        if definition.description:
            return definition.description
        lexicon = definition.keywords or PROFILE_KEYWORDS.get(name, ())
        if lexicon:
            return f"{name} routing profile: {', '.join(lexicon)}"
        return f"{name} routing profile"

    async def _ensure_vectors(self, embedder) -> dict[str, list[float]]:
        if self._profile_vectors is not None:
            return self._profile_vectors
        profiles = self._service.config.profiles
        texts = [self._description_for(name, d) for name, d in profiles.items()]
        vectors = await embedder.embed(texts)
        self._profile_vectors = dict(zip(profiles.keys(), vectors, strict=True))
        return self._profile_vectors

    async def infer_profile(self, query: str | None) -> tuple[str | None, str]:
        """Classify a query. Returns (profile, method).

        method is one of 'embedding', 'keyword', or 'none' so callers and
        metrics can see which mechanism actually decided.
        """
        if not query:
            return None, "none"

        if time.monotonic() >= self._cooldown_until:
            embedder = self._embedder_factory()
            if embedder is not None:
                try:
                    vectors = await self._ensure_vectors(embedder)
                    [query_vector] = await embedder.embed([query])
                    best_name: str | None = None
                    best_score = -1.0
                    for profile_name, vector in vectors.items():
                        score = cosine_similarity(query_vector, vector)
                        if score > best_score:
                            best_name, best_score = profile_name, score
                    threshold = settings.router_embedding_threshold
                    if best_name is not None and best_score >= threshold:
                        logger.info({
                            "message": "Semantic router matched via embeddings",
                            "profile": best_name,
                            "score": round(best_score, 4),
                        })
                        return best_name, "embedding"
                except Exception as exc:  # noqa: BLE001 - degrade, never fail the request
                    self._cooldown_until = time.monotonic() + self._cooldown_s
                    logger.warning({
                        "message": "Embedding router unavailable; using keywords",
                        "error": str(exc),
                    })

        matched = self._service.infer_profile(query)
        return (matched, "keyword") if matched else (None, "none")


semantic_router = SemanticRouter(binding_service)
