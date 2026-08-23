import json

import pytest
from fastapi.testclient import TestClient

from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app


class StreamStubConnector(BaseConnector):
    is_available = True

    def __init__(self, connector_id: str):
        self.connector_id = connector_id
        self.display_name = f"{connector_id.title()} Stub"
        self.capabilities = ["text"]

    async def query(self, prompt, sub_query, config):
        if '"role": "researcher"' in sub_query:
            content = '{"facts":["f"],"constraints":[],"references":[],"unknowns":[],"confidence":"high"}'
        elif '"role": "analyzer"' in sub_query:
            content = '{"proposed_solution":"sol","assumptions":[],"tradeoffs":[],"risks":[],"validation_checks":[]}'
        elif '"role": "verifier"' in sub_query:
            content = '{"critical_risks":[],"hidden_assumptions":[],"edge_cases":[],"validation_requirements":[]}'
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


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data_line = next((line[6:] for line in lines if line.startswith("data: ")), "{}")
        events.append((name, json.loads(data_line)))
    return events


@pytest.fixture
def stream_client(monkeypatch):
    stubs = {"mistral": StreamStubConnector("mistral"), "gemini": StreamStubConnector("gemini")}
    with TestClient(app) as test_client:
        monkeypatch.setattr(registry, "_connectors", stubs)
        yield test_client


def test_stream_parallel_path_emits_full_sequence(stream_client):
    query = " ".join(["Explain the ARGUS architecture and compare latency tradeoffs."] * 10)
    with stream_client.stream(
        "POST", "/v1/query/stream", json={"query": query}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())

    events = _parse_sse(raw)
    names = [name for name, _ in events]

    role_events = [e for n, e in events if n == "role_complete"]
    assert sorted(e["role"] for e in role_events) == ["analyzer", "researcher", "verifier"]
    assert all(e["status"] == "success" for e in role_events)

    final = next(e for n, e in events if n == "final")
    assert names[-1] == "final"
    assert final["result"] == "Synthesized response"
    assert final["short_circuited"] is False
    assert set(final["role_assignments"]) >= {"synthesizer"}
    assert len(final["model_statuses"]) == 3


def test_stream_short_query_direct_path(stream_client):
    with stream_client.stream(
        "POST", "/v1/query/stream", json={"query": "What is ARGUS?"}
    ) as response:
        raw = "".join(response.iter_text())

    events = _parse_sse(raw)
    names = [name for name, _ in events]
    assert names.count("role_complete") == 1
    assert events[0][1]["role"] == "direct"
    assert names[-1] == "final"
    final = events[-1][1]
    assert final["short_circuited"] is True
    assert final["result"] == "Direct response"


def test_stream_unknown_connector_422(stream_client):
    response = stream_client.post(
        "/v1/query/stream",
        json={"query": "What is ARGUS?", "model_config": {"connectors": ["bogus"]}},
    )
    assert response.status_code == 422
