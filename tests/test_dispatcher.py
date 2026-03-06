import pytest
from app.orchestration.dispatcher import dispatch
from app.connectors.base import ConnectorConfig, ConnectorStatus

@pytest.mark.asyncio
async def test_dispatch_happy_path(mock_connector):
    sub_queries = {"mock": "What is Python?"}
    connectors = {"mock": mock_connector}
    result = await dispatch("Original query", sub_queries, connectors)
    assert "mock" in result
    assert result["mock"].status == ConnectorStatus.SUCCESS

@pytest.mark.asyncio
async def test_dispatch_partial_failure(mock_connector, failing_connector):
    sub_queries = {"mock": "task 1", "fail": "task 2"}
    failing_connector.connector_id = "fail"
    connectors = {"mock": mock_connector, "fail": failing_connector}
    result = await dispatch("original", sub_queries, connectors)
    assert result["mock"].status == ConnectorStatus.SUCCESS
    assert result["fail"].status == ConnectorStatus.ERROR
@pytest.mark.asyncio
async def test_dispatch_timeout(timeout_connector):
    sub_queries = {"mock": "task"}
    timeout_connector.connector_id = "mock"
    connectors = {"mock": timeout_connector}
    result = await dispatch("original", sub_queries, connectors)
    assert result["mock"].status == ConnectorStatus.TIMEOUT
@pytest.mark.asyncio
async def test_dispatch_empty():
    result = await dispatch("query", {}, {})
    assert result == {}