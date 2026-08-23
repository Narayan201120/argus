import pytest

from app.connectors.base import ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.orchestration.contracts import SharedTaskState
from app.orchestration.workers import _query_with_retry

QUERY = "What is Python?"
SUB_QUERY = "collect facts"


class ScriptedConnector:
    """Minimal stand-in returning queued responses per call."""

    connector_id = "scripted"

    def __init__(self, *responses: ConnectorResponse):
        self._responses = list(responses)
        self.calls = 0

    async def query(self, prompt, sub_query, config):
        self.calls += 1
        return self._responses.pop(0)


def _response(status: ConnectorStatus) -> ConnectorResponse:
    return ConnectorResponse(
        model_id="scripted-model",
        content="{}" if status == ConnectorStatus.SUCCESS else "",
        latency_ms=5,
        token_usage=TokenUsage(),
        status=status,
        sub_query=SUB_QUERY,
    )


def _shared_state() -> SharedTaskState:
    return SharedTaskState(
        request_id="req-1",
        original_query=QUERY,
        main_objective=QUERY,
        expected_final_output="json",
    )


@pytest.mark.asyncio
async def test_retry_returns_success_without_second_call():
    connector = ScriptedConnector(_response(ConnectorStatus.SUCCESS))
    response = await _query_with_retry(
        connector, QUERY, SUB_QUERY, ConnectorConfig(max_retries=1)
    )
    assert response.status == ConnectorStatus.SUCCESS
    assert connector.calls == 1


@pytest.mark.asyncio
async def test_timeout_retries_once_then_succeeds():
    connector = ScriptedConnector(
        _response(ConnectorStatus.TIMEOUT),
        _response(ConnectorStatus.SUCCESS),
    )
    response = await _query_with_retry(
        connector, QUERY, SUB_QUERY, ConnectorConfig(max_retries=1)
    )
    assert response.status == ConnectorStatus.SUCCESS
    assert connector.calls == 2


@pytest.mark.asyncio
async def test_timeout_exhausts_retries():
    connector = ScriptedConnector(
        _response(ConnectorStatus.TIMEOUT),
        _response(ConnectorStatus.TIMEOUT),
    )
    response = await _query_with_retry(
        connector, QUERY, SUB_QUERY, ConnectorConfig(max_retries=1)
    )
    assert response.status == ConnectorStatus.TIMEOUT
    assert connector.calls == 2


@pytest.mark.asyncio
async def test_no_retry_when_max_retries_zero():
    connector = ScriptedConnector(_response(ConnectorStatus.TIMEOUT))
    response = await _query_with_retry(
        connector, QUERY, SUB_QUERY, ConnectorConfig(max_retries=0)
    )
    assert response.status == ConnectorStatus.TIMEOUT
    assert connector.calls == 1


@pytest.mark.asyncio
async def test_error_is_not_retried():
    connector = ScriptedConnector(_response(ConnectorStatus.ERROR))
    response = await _query_with_retry(
        connector, QUERY, SUB_QUERY, ConnectorConfig(max_retries=1)
    )
    assert response.status == ConnectorStatus.ERROR
    assert connector.calls == 1
