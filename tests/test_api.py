from fastapi.testclient import TestClient

from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app

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


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "ARGUS" in response.json()["name"]


def test_health():
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connectors" in data


def test_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "connectors" in data
    assert "total" in data
    assert data["total"] >= 0


def test_query_rejects_unknown_connectors():
    response = client.post("/v1/query", json={
        "query": "test",
        "model_config": {"connectors": ["nonexistent"]},
    })
    assert response.status_code == 422
    assert "Unknown connector IDs" in response.json()["detail"]


def test_short_query_uses_requested_connector(monkeypatch):
    mistral = StubConnector("mistral")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral})

    response = client.post("/v1/query", json={
        "query": "What is ARGUS?",
        "model_config": {"connectors": ["mistral"]},
    })

    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "Direct response"
    assert data["synthesizer"] == "mistral"
    assert data["short_circuited"] is True
    assert data["model_statuses"] == [{
        "role": "direct",
        "connector_id": "mistral",
        "status": "success",
        "latency_ms": 1,
        "error": None,
        "token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "sub_query": "What is ARGUS?",
        "retry_after_s": None,
    }]


def test_parallel_query_stays_with_requested_connector(monkeypatch):
    mistral = StubConnector("mistral")
    gemini = StubConnector("gemini")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral, "gemini": gemini})
    query = " ".join(["Explain the ARGUS architecture and compare its latency tradeoffs."] * 10)

    response = client.post("/v1/query", json={
        "query": query,
        "model_config": {"connectors": ["mistral"]},
    })

    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "Synthesized response"
    assert data["short_circuited"] is False
    assert {status["role"] for status in data["model_statuses"]} == {
        "researcher", "analyzer", "verifier",
    }
    assert {status["connector_id"] for status in data["model_statuses"]} == {"mistral"}

def test_startup_registers_all_connector_implementations():
    with TestClient(app) as app_client:
        response = app_client.get("/v1/models")

    assert response.status_code == 200
    assert {profile["connector_id"] for profile in response.json()["connectors"]} == {
        "gemini", "openai", "claude", "mistral",
    }


def test_parallel_query_reports_role_assignments(monkeypatch):
    mistral = StubConnector("mistral")
    gemini = StubConnector("gemini")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral, "gemini": gemini})
    query = " ".join(["Explain the ARGUS architecture and compare its latency tradeoffs."] * 10)

    response = client.post("/v1/query", json={"query": query})

    assert response.status_code == 200
    assignments = response.json()["role_assignments"]
    assert set(assignments) >= {"researcher", "analyzer", "verifier", "synthesizer"}
    assert all(a in {"mistral", "gemini"} for a in assignments.values())


def test_role_binding_override_is_honored(monkeypatch):
    mistral = StubConnector("mistral")
    gemini = StubConnector("gemini")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral, "gemini": gemini})
    query = " ".join(["Explain the ARGUS architecture and compare its latency tradeoffs."] * 10)

    response = client.post("/v1/query", json={
        "query": query,
        "model_config": {
            "role_bindings": {"researcher": ["mistral"], "synthesizer": ["mistral"]},
        },
    })

    assert response.status_code == 200
    data = response.json()
    assert data["role_assignments"]["researcher"] == "mistral"
    assert data["role_assignments"]["synthesizer"] == "mistral"


def test_unknown_role_binding_rejected(monkeypatch):
    mistral = StubConnector("mistral")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral})

    response = client.post("/v1/query", json={
        "query": "What is ARGUS?",
        "model_config": {"role_bindings": {"captain": ["mistral"]}},
    })

    assert response.status_code == 422
    assert "Unknown roles in role_bindings" in response.json()["detail"]


def test_unknown_profile_rejected():
    response = client.post("/v1/query", json={
        "query": "What is ARGUS?",
        "model_config": {"profile": "bogus_profile"},
    })
    assert response.status_code == 422
    assert "Unknown profile" in response.json()["detail"]


def test_known_profile_restricts_connectors(monkeypatch):
    mistral = StubConnector("mistral")
    gemini = StubConnector("gemini")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral, "gemini": gemini})

    response = client.post("/v1/query", json={
        "query": "What is ARGUS?",
        "model_config": {"profile": "fast"},
    })

    assert response.status_code == 200
    data = response.json()
    assert data["short_circuited"] is True
    assert data["model_statuses"][0]["connector_id"] in {"mistral", "gemini"}


def test_query_cache_hit_on_second_identical_request(monkeypatch):
    from fakeredis import aioredis as fakeredis_aioredis

    from app.cache import ResponseCache
    from app.rediskit import holder as redis_holder

    mistral = StubConnector("mistral")
    monkeypatch.setattr(registry, "_connectors", {"mistral": mistral})
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_holder, "client", fake_redis)
    monkeypatch.setattr(redis_holder, "cache", ResponseCache(fake_redis))

    payload = {
        "query": "Explain the ARGUS architecture in depth.",
        "model_config": {"connectors": ["mistral"]},
    }
    first = client.post("/v1/query", json=payload)
    second = client.post("/v1/query", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert second.json()["result"] == first.json()["result"]
    assert second.json()["role_assignments"] == first.json()["role_assignments"]
