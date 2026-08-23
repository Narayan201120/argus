"""Stage 7 - Prometheus /v1/metrics endpoint and instrumentation."""

from fakeredis import aioredis as fakeredis_aioredis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app
from app.ratelimit import RateLimitMiddleware
from app.rediskit import holder as redis_holder

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
            token_usage=TokenUsage(3, 5, 8),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


def _scrape() -> bytes:
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return response.content


def test_metrics_endpoint_exposes_registry():
    body = _scrape()
    assert b"argus_http_requests_total" in body
    assert b"argus_role_outcomes_total" in body
    assert b"# TYPE argus_http_in_flight gauge" in body


def test_query_records_http_and_role_metrics(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {"mistral": StubConnector("mistral")})
    long_query = " ".join(
        ["Explain the ARGUS architecture and compare its latency tradeoffs."] * 10
    )
    response = client.post("/v1/query", json={"query": long_query})
    assert response.status_code == 200

    body = _scrape()
    assert b'argus_http_requests_total{method="POST",path="/v1/query"' in body
    assert b'argus_role_outcomes_total{connector_id="mistral"' in body
    assert b'role="researcher"' in body
    assert b'argus_role_tokens_total{connector_id="mistral",role="researcher",type="prompt"' in body
    assert b'argus_role_latency_seconds_count{connector_id="mistral"' in body


def test_direct_path_records_direct_role(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {"mistral": StubConnector("mistral")})
    response = client.post("/v1/query", json={"query": "Hi"})
    assert response.status_code == 200

    body = _scrape()
    assert b'argus_role_outcomes_total{connector_id="mistral",role="direct",status="success"' in body


def test_cache_operations_counter(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {"mistral": StubConnector("mistral")})
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_holder, "client", fake_redis)

    payload = {"query": "Explain the ARGUS architecture.", "model_config": {"connectors": ["mistral"]}}
    first = client.post("/v1/query", json=payload)
    second = client.post("/v1/query", json=payload)
    assert first.status_code == second.status_code == 200

    body = _scrape()
    assert b'argus_cache_operations_total{result="hit"}' in body
    assert b'argus_cache_operations_total{result="miss"}' in body


def test_rate_limit_rejection_counter(monkeypatch):
    mini = FastAPI()

    @mini.get("/ping")
    async def ping():
        return {"ok": True}

    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_holder, "client", fake_redis)
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_max_requests", 1)

    scoped = TestClient(RateLimitMiddleware(mini, redis_holder))
    assert scoped.get("/ping").status_code == 200
    assert scoped.get("/ping").status_code == 429

    assert b"argus_rate_limit_rejections_total" in _scrape()


def test_metrics_exempt_from_auth(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "x" * 32)

    assert client.post("/v1/query", json={"query": "Hi"}).status_code == 401
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/metrics").status_code == 200


def test_metrics_exempt_from_rate_limit(monkeypatch):
    fake_redis = fakeredis_aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_holder, "client", fake_redis)
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_max_requests", 1)

    assert client.get("/v1/metrics").status_code == 200
    assert client.get("/v1/metrics").status_code == 200
