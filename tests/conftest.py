import pytest

from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage


class MockConnector(BaseConnector):
    connector_id = "mock"
    display_name = "Mock Connector"
    capabilities = ["text", "code"]
    is_available = True

    def __init__(self, status=ConnectorStatus.SUCCESS, content="Mock response"):
        self._status = status
        self._content = content

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id = "mock-model",
            content = self._content if self._status == ConnectorStatus.SUCCESS else "",
            latency_ms = 100,
            token_usage = TokenUsage(10, 20, 30),
            status = self._status,
            error = None if self._status == ConnectorStatus.SUCCESS else "Mock Error",
            sub_query = sub_query,
        )
    async def health_check(self):
        return self.is_available

@pytest.fixture
def mock_connector():
    return MockConnector()

@pytest.fixture
def failing_connector():
    return MockConnector(status=ConnectorStatus.ERROR)

@pytest.fixture
def timeout_connector():
    return MockConnector(status=ConnectorStatus.TIMEOUT)
