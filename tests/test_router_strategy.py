"""Stage 6 - router strategy selection (static vs semantic)."""

from fastapi.testclient import TestClient

from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app
from app.orchestration.binding import ROUTER_STRATEGIES, RoleBindingService

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
