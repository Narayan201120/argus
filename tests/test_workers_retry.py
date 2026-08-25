import pytest

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.orchestration.workers import RoleTaskError, run_connector_query


class CountingConnector(BaseConnector):
    connector_id = "counting"
    display_name = "Counting"
    capabilities = ["text"]
    is_available = True

    def __init__(self, status=ConnectorStatus.SUCCESS):
        self.status = status
        self.calls = 0

    async def query(self, prompt, sub_query, config):
        self.calls += 1
        return ConnectorResponse(
            model_id="stub-model",
            content="" if self.status != ConnectorStatus.SUCCESS else "ok",
            latency_ms=5,
            token_usage=TokenUsage(1, 1, 2),
            status=self.status,
            error=None if self.status == ConnectorStatus.SUCCESS else "provider issue",
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_success_passthrough():
    connector = CountingConnector()
    response = await run_connector_query(connector, "p", "s", ConnectorConfig())
    assert response.status == ConnectorStatus.SUCCESS
    assert connector.calls == 1


@pytest.mark.asyncio
async def test_timeout_is_not_retried_same_provider():
    """Timeout consumes the whole budget: no same-provider second attempt."""
    connector = CountingConnector(status=ConnectorStatus.TIMEOUT)
    response = await run_connector_query(connector, "p", "s", ConnectorConfig())
    assert response.status == ConnectorStatus.TIMEOUT
    assert connector.calls == 1  # exactly one attempt, no doubling


def test_role_task_error_preserves_response_and_hint():
    response = ConnectorResponse(
        model_id="gemini",
        content="",
        latency_ms=40,
        token_usage=TokenUsage(),
        status=ConnectorStatus.RATE_LIMITED,
        error="429 You exceeded your current quota.",
        retry_after_s=8.0,
    )
    error = RoleTaskError(role="researcher", response=response)

    assert error.role == "researcher"
    assert error.response.status == ConnectorStatus.RATE_LIMITED
    assert error.response.retry_after_s == 8.0
    assert "provider rate limited" in str(error)
    assert "retry_after_s=8.0" in str(error)
