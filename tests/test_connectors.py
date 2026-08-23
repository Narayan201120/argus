import pytest

from app.connectors.base import (
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    classify_provider_exception,
)
from app.connectors.gemini import GeminiConnector
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


def test_classify_detects_429_status_code():
    exc = TypeError("blocked")
    exc.status_code = 429  # type: ignore[attr-defined]
    status, _ = classify_provider_exception(exc)
    assert status == ConnectorStatus.RATE_LIMITED


def test_classify_detects_rate_limit_error_type():
    class OpenAIRateLimitError(Exception):
        pass

    status, _ = classify_provider_exception(OpenAIRateLimitError("Too many requests"))
    assert status == ConnectorStatus.RATE_LIMITED


def test_classify_detects_quota_message_and_parses_nothing():
    status, retry_after_s = classify_provider_exception(
        Exception("429 You exceeded your current quota ... GenerateRequestsPerMinutePerProjectPerModel")
    )
    assert status == ConnectorStatus.RATE_LIMITED
    assert retry_after_s is None


def test_classify_extracts_retry_after_header():
    from types import SimpleNamespace

    exc = Exception("slow down")
    exc.response = SimpleNamespace(headers={"retry-after": "12"})  # type: ignore[attr-defined]
    exc.status_code = 429  # type: ignore[attr-defined]
    status, retry_after_s = classify_provider_exception(exc)
    assert status == ConnectorStatus.RATE_LIMITED
    assert retry_after_s == 12.0


def test_classify_plain_error_stays_error():
    status, retry_after_s = classify_provider_exception(ValueError("bad json"))
    assert status == ConnectorStatus.ERROR
    assert retry_after_s is None


@pytest.mark.asyncio
async def test_gemini_maps_quota_error_to_rate_limited(monkeypatch):
    from types import SimpleNamespace

    connector = GeminiConnector()
    connector.api_key = "test-key"
    connector.is_available = True

    quota_error = Exception("429 Resource has been exhausted (quota exceeded).")
    fake_models = SimpleNamespace(generate_content=lambda **kwargs: (_ for _ in ()).throw(quota_error))
    connector._client = SimpleNamespace(models=fake_models)

    response = await connector.query("prompt", "sub", ConnectorConfig())
    assert response.status == ConnectorStatus.RATE_LIMITED
    assert "quota" in (response.error or "")
