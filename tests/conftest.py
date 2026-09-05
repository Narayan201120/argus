import pytest

from app.analysis import workers as analysis_workers
from app.analysis.workers import AnalysisOutput, CritiqueOutput, GapOutput
from app.config import settings
from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage


@pytest.fixture(autouse=True)
def _no_live_embeddings(monkeypatch):
    """Mock-only discipline (DEC-008): tests never call real embedding
    APIs even when local .env has provider keys. Embedding-specific tests
    override this by setting the provider themselves."""
    monkeypatch.setattr(settings, "router_embedding_provider", "none")


@pytest.fixture(autouse=True)
def _no_live_analysis_workers(monkeypatch):
    """Mock-only discipline (DEC-008): the investigation loop's LLM workers
    never fire in tests, even when local .env has provider keys. Scripted
    defaults: no claims, no challenges, sufficient immediately. Tests needing
    specific worker behavior monkeypatch over this fixture."""
    async def _analyze(board_text: str, query: str) -> AnalysisOutput:
        return AnalysisOutput(claims=[])

    async def _critique(board_text: str, query: str) -> CritiqueOutput:
        return CritiqueOutput(challenges=[])

    async def _assess(board_text: str, query: str) -> GapOutput:
        return GapOutput(sufficient=True, rationale="test stub: sufficient")

    monkeypatch.setattr(analysis_workers, "analyze_board", _analyze)
    monkeypatch.setattr(analysis_workers, "critique_board", _critique)
    monkeypatch.setattr(analysis_workers, "assess_gaps", _assess)


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
