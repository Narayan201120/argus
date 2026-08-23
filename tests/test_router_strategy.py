"""Stage 6/P2-1 - router strategy selection (static vs semantic)."""

import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app
from app.orchestration.binding import (
    ROUTER_STRATEGIES,
    RoleBindingService,
    SemanticRouter,
)

client = TestClient(app)


class StubConnector(BaseConnector):
    capabilities = ["text"]
    is_available = True

    def __init__(self, connector_id: str):
        self.connector_id = connector_id
        self.display_name = f"{connector_id.title()} Stub"

    async def query(self, prompt, sub_query, config):
        if '"role": "researcher"' in sub_query:
            content = '{"facts":["Fact"],"constraints":[],"references":[],"unknowns":[],"confidence":"high"}'
        elif '"role": "analyzer"' in sub_query:
            content = '{"proposed_solution":"Use the findings.","assumptions":[],"tradeoffs":[],"risks":[],"validation_checks":[]}'
        elif '"role": "verifier"' in sub_query:
            content = '{"critical_risks":[],"hidden_assumptions":[],"edge_cases":[],"validation_requirements":[],"confidence":"high"}'
        elif "synthesis layer" in prompt:
            content = "Synthesized response"
        else:
            content = "Direct response"

        return ConnectorResponse(
            model_id=self.connector_id,
            content=content,
            latency_ms=1,
            token_usage=TokenUsage(1, 1, 2),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


def test_router_strategy_registry_is_static_and_semantic():
    assert set(ROUTER_STRATEGIES) == {"static", "semantic"}
    assert RoleBindingService.load().known_strategy("static")
    assert RoleBindingService.load().known_strategy("semantic")
    assert not RoleBindingService.load().known_strategy("bogus")


def test_infer_profile_matches_research_intent():
    service = RoleBindingService.load()
    assert service.infer_profile("Please compare the frameworks and survey the literature") == "research"


def test_infer_profile_matches_code_intent():
    service = RoleBindingService.load()
    assert service.infer_profile("Fix this Python bug inside the function") == "code"


def test_infer_profile_matches_analysis_intent():
    service = RoleBindingService.load()
    assert service.infer_profile("Evaluate the tradeoffs and risks of this architecture decision") == "analysis"


def test_infer_profile_is_case_insensitive():
    service = RoleBindingService.load()
    assert service.infer_profile("Give me a QUICK SHORT ANSWER") == "fast"


def test_infer_profile_returns_none_without_match():
    service = RoleBindingService.load()
    assert service.infer_profile("Hello there") is None
    assert service.infer_profile("") is None
    assert service.infer_profile(None) is None


def test_infer_profile_skips_profiles_without_lexicon(tmp_path):
    routing_file = tmp_path / "routing.yaml"
    routing_file.write_text(
        "roles:\n  researcher: [gemini]\nprofiles:\n  legal: [gemini]\n",
        encoding="utf-8",
    )
    service = RoleBindingService.load(str(routing_file))
    # YAML merges over defaults, so 'fast' stays configured and inferable
    assert service.known_profile("fast") is True
    assert service.infer_profile("Give me a QUICK SHORT ANSWER") == "fast"
    # custom profiles have no lexicon entry and are never inferred
    assert service.known_profile("legal") is True
    assert service.infer_profile("legal contracts question") is None


def test_unknown_router_strategy_rejected_with_422(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {"mistral": StubConnector("mistral")})
    response = client.post("/v1/query", json={
        "query": "What is ARGUS?",
        "model_config": {"router_strategy": "bogus"},
    })
    assert response.status_code == 422
    assert "Unknown router strategy: bogus" in response.json()["detail"]


def test_semantic_strategy_infers_fast_pool_for_short_query(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {
        "gemini": StubConnector("gemini"),
        "openai": StubConnector("openai"),
        "claude": StubConnector("claude"),
        "mistral": StubConnector("mistral"),
    })
    response = client.post("/v1/query", json={
        "query": "quick short answer about ARGUS",
        "model_config": {"router_strategy": "semantic"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["router_strategy"] == "semantic"
    assert data["matched_profile"] == "fast"
    # fast pool = [mistral, gemini]; direct chain picks gemini within it
    assert data["short_circuited"] is True
    assert data["role_assignments"]["direct"] == "gemini"


def test_static_strategy_keeps_default_chains(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {
        "gemini": StubConnector("gemini"),
        "openai": StubConnector("openai"),
        "claude": StubConnector("claude"),
        "mistral": StubConnector("mistral"),
    })
    response = client.post("/v1/query", json={
        "query": "quick short answer about ARGUS",
        "model_config": {"router_strategy": "static"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["router_strategy"] == "static"
    assert data["matched_profile"] is None
    # unrestricted pool: direct chain starts claude -> openai -> ...
    assert data["role_assignments"]["direct"] == "claude"


def test_explicit_profile_wins_over_semantic_inference(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {
        "gemini": StubConnector("gemini"),
        "openai": StubConnector("openai"),
        "claude": StubConnector("claude"),
        "mistral": StubConnector("mistral"),
    })
    response = client.post("/v1/query", json={
        "query": "quick short answer about ARGUS",
        "model_config": {"profile": "research", "router_strategy": "semantic"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["router_strategy"] == "semantic"
    assert data["matched_profile"] is None
    # explicit research pool honored; inference suppressed
    assert data["role_assignments"]["direct"] == "claude"


# ── Stage P2-1: embeddings-first semantic router ────────────────────────────


class ScriptedEmbedder:
    """Returns one pre-scripted batch of vectors per embed() call."""

    def __init__(self, batches: list[list[list[float]]], fail: bool = False):
        self._batches = batches
        self._fail = fail
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        if self._fail:
            raise RuntimeError("embedding provider down")
        batch = self._batches[self.calls - 1]
        assert len(batch) == len(texts), f"scripted {len(batch)} vectors, asked {len(texts)}"
        return [list(vector) for vector in batch]


REAL_PROFILE_ORDER = ["research", "code", "analysis", "fast"]


def _unit_batches_for(query_vector: list[float]) -> list[list[list[float]]]:
    unit = {
        "research": [1.0, 0.0, 0.0, 0.0],
        "code": [0.0, 1.0, 0.0, 0.0],
        "analysis": [0.0, 0.0, 1.0, 0.0],
        "fast": [0.0, 0.0, 0.0, 1.0],
    }
    warmup = [unit[name] for name in REAL_PROFILE_ORDER]
    return [warmup, [query_vector]]


@pytest.fixture
def router_with_real_profiles():
    """Fresh SemanticRouter over the real routing config; no vectors yet."""
    return SemanticRouter(RoleBindingService.load(), embedder_factory=lambda: None)


@pytest.mark.asyncio
async def test_semantic_router_empty_query_skips_embeddings(router_with_real_profiles):
    profile, method = await router_with_real_profiles.infer_profile("")
    assert profile is None
    assert method == "none"


@pytest.mark.asyncio
async def test_embedding_match_above_threshold(monkeypatch):
    # Query vector identical to research's unit vector -> cosine 1.0.
    fake = ScriptedEmbedder(_unit_batches_for([1.0, 0.0, 0.0, 0.0]))
    router = SemanticRouter(RoleBindingService.load(), embedder_factory=lambda: fake)

    monkeypatch.setattr(settings, "router_embedding_threshold", 0.9)
    profile, method = await router.infer_profile("survey prior work on retrieval")
    assert (profile, method) == ("research", "embedding")


@pytest.mark.asyncio
async def test_below_threshold_falls_back_to_keywords(monkeypatch):
    # Query leans toward research (0.6) but the threshold is 0.9, so the
    # embedding match is rejected and the keyword classifier decides:
    # 'quick'/'short answer' hits the fast lexicon.
    fake = ScriptedEmbedder(_unit_batches_for([0.6, 0.8, 0.0, 0.0]))
    router = SemanticRouter(RoleBindingService.load(), embedder_factory=lambda: fake)

    monkeypatch.setattr(settings, "router_embedding_threshold", 0.9)
    profile, method = await router.infer_profile("quick short answer about ARGUS")
    assert (profile, method) == ("fast", "keyword")


@pytest.mark.asyncio
async def test_embedder_failure_starts_cooldown_and_uses_keywords():
    failing = ScriptedEmbedder([], fail=True)
    router = SemanticRouter(
        RoleBindingService.load(), embedder_factory=lambda: failing, cooldown_s=60.0
    )

    profile, method = await router.infer_profile("fix this python bug in my function")
    assert method == "keyword"
    assert profile == "code"
    assert router._cooldown_until > time.monotonic()

    # Within the cooldown the factory is not even invoked
    profile2, method2 = await router.infer_profile("another python bug")
    assert failing.calls == 1
    assert method2 == "keyword"


@pytest.mark.asyncio
async def test_router_decision_metric_recorded(monkeypatch):
    from prometheus_client import REGISTRY

    from app.api.routes import shared as shared_module

    # Embedding match rejected (0.6 < 0.9) -> keyword decides -> fast.
    fake = ScriptedEmbedder(_unit_batches_for([0.6, 0.8, 0.0, 0.0]))
    fresh = SemanticRouter(RoleBindingService.load(), embedder_factory=lambda: fake)
    monkeypatch.setattr(shared_module, "semantic_router", fresh)
    monkeypatch.setattr(settings, "router_embedding_threshold", 0.9)
    monkeypatch.setattr(registry, "_connectors", {
        "gemini": StubConnector("gemini"),
        "openai": StubConnector("openai"),
        "claude": StubConnector("claude"),
        "mistral": StubConnector("mistral"),
    })

    response = client.post("/v1/query", json={
        "query": "quick short answer about ARGUS",
        "model_config": {"router_strategy": "semantic"},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["matched_profile"] == "fast"
    assert data["router_strategy"] == "semantic"

    value = REGISTRY.get_sample_value(
        "argus_router_decisions_total",
        {"method": "keyword", "matched_profile": "fast"},
    )
    assert value is not None and value >= 1.0


def test_cache_keys_distinguish_router_strategies(monkeypatch):
    from fakeredis import aioredis as fakeredis_aioredis

    from app.cache import ResponseCache
    from app.rediskit import holder as redis_holder

    monkeypatch.setattr(registry, "_connectors", {"mistral": StubConnector("mistral")})
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_holder, "client", fake_redis)
    monkeypatch.setattr(redis_holder, "cache", ResponseCache(fake_redis))

    # Neutral query: no profile inference, so every pool resolves to the
    # single registered connector regardless of strategy.
    base = {"query": "Hello there"}
    semantic_first = client.post("/v1/query", json={
        **base, "model_config": {"router_strategy": "semantic"},
    })
    semantic_second = client.post("/v1/query", json={
        **base, "model_config": {"router_strategy": "semantic"},
    })
    static_request = client.post("/v1/query", json={
        **base, "model_config": {"router_strategy": "static"},
    })

    assert semantic_first.status_code == semantic_second.status_code == 200
    assert semantic_second.json()["cache_hit"] is True
    assert static_request.status_code == 200
    assert static_request.json()["cache_hit"] is False
