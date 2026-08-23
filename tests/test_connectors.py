import pytest

from app.connectors.base import ConnectorConfig, ConnectorResponse, ConnectorStatus
from app.connectors.registry import ConnectorRegistry


@pytest.mark.asyncio
async def test_connector_response_shape(mock_connector):
    query = await mock_connector.query("test", "test", ConnectorConfig())
    assert isinstance(query, ConnectorResponse)
    assert query.model_id == "mock-model"
    assert query.status == ConnectorStatus.SUCCESS
    assert query.token_usage is not None

def test_registry_register_and_get(mock_connector):
    reg = ConnectorRegistry()
    reg.register(mock_connector)
    assert reg.get("mock") is mock_connector

def test_registry_available(mock_connector):
    reg = ConnectorRegistry()
    reg.register(mock_connector)
    assert len(reg.available()) == 1

def test_capability_profile(mock_connector):
    profile = mock_connector.capability_profile()
    assert "connector_id" in profile

@pytest.mark.asyncio
async def test_error_connector(failing_connector):
    response = await failing_connector.query("test", "test", ConnectorConfig())
    assert response.status == ConnectorStatus.ERROR
    assert response.error is not None
    assert response.content == ""
