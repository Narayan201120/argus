import time

import pytest
from fastapi.testclient import TestClient

from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app


class ReportStubConnector(BaseConnector):
    """Routes by marker text in the prompt/sub_query, like live role prompts."""

    is_available = True

    def __init__(self, connector_id: str):
        self.connector_id = connector_id
        self.display_name = f"{connector_id.title()} Stub"
        self.capabilities = ["text"]
        self.writer_calls = 0
        self.reviewer_verdicts: list[str] = []

    async def query(self, prompt, sub_query, config):
        if "report planner" in prompt or "report planner" in sub_query:
            content = (
                '{"report_title": "Demo Report", "subtasks": ['
                '{"subtask_id": "s1", "title": "Track One", "objective": "Investigate one"},'
                '{"subtask_id": "s2", "title": "Track Two", "objective": "Investigate two"}'
                "]}"
            )
        elif '"role": "researcher"' in sub_query:
            content = '{"facts":["fact-1"],"constraints":[],"references":[],"unknowns":[],"confidence":"high"}'
        elif '"role": "analyzer"' in sub_query:
            content = '{"proposed_solution":"Use the findings.","assumptions":[],"tradeoffs":[],"risks":[],"validation_checks":[]}'
        elif '"role": "verifier"' in sub_query:
            content = '{"critical_risks":["risk-1"],"hidden_assumptions":[],"edge_cases":[],"validation_requirements":[]}'
        elif "report writer" in prompt:
            self.writer_calls += 1
            section = (
                "\n## Review Fix Section\n\nAddressed issues."
                if self.reviewer_verdicts and self.reviewer_verdicts[-1] == "reject"
                else ""
            )
            content = "# Demo Report\n\n## Executive Summary\n\nAll good." + section
        elif "report reviewer" in prompt:
            rejected = self.writer_calls == 1 and len(self.reviewer_verdicts) == 0
            self.reviewer_verdicts.append("reject" if rejected else "approve")
            content = (
                '{"approved": false, "issues": ["Missing coverage of track two"]}'
                if rejected
                else '{"approved": true, "issues": []}'
            )
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


@pytest.fixture
def report_client(monkeypatch):
    stub = ReportStubConnector("mistral")
    with TestClient(app) as test_client:
        # Patch AFTER lifespan so startup-registered real connectors are
        # replaced wholesale; tests must never reach live providers.
        monkeypatch.setattr(registry, "_connectors", {"mistral": stub})
        yield test_client, stub


def test_report_full_flow_with_repair_round(report_client):
    client, stub = report_client

    created = client.post("/v1/report", json={"query": "Research the ARGUS pipeline deeply"})
    assert created.status_code == 202
    body = created.json()
    assert body["status"] in {"queued", "running"}
    assert body["poll_url"] == f"/v1/report/{body['job_id']}"

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status_response = client.get(body["poll_url"])
        assert status_response.status_code == 200
        snapshot = status_response.json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert snapshot["status"] == "completed", snapshot.get("error")
    assert snapshot["result_markdown"] is not None
    assert "# Demo Report" in snapshot["result_markdown"]
    assert snapshot["role_assignments"]["planner"] == "mistral"
    assert snapshot["role_assignments"]["writer"] == "mistral"
    # reviewer rejected the first draft once -> writer ran twice
    assert stub.writer_calls == 2
    assert snapshot["role_assignments"].get("reviewer") == "mistral"


def test_get_unknown_report_job_404(report_client):
    client, _ = report_client
    response = client.get("/v1/report/does-not-exist")
    assert response.status_code == 404
