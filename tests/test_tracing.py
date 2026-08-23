"""Stage P2-3 - opt-in OpenTelemetry tracing (mock-only)."""

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from app.config import settings
from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import ConnectorRegistry
from app.tracing import configure_tracing, span


class ListSpanExporter(SpanExporter):
    """In-memory exporter: deterministic assertions without stdout races."""

    def __init__(self):
        self.spans: list = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class StubConnector(BaseConnector):
    connector_id = "stub"
    display_name = "Stub"
    capabilities = ["text"]
    is_available = True

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id="stub-model",
            content="ok",
            latency_ms=3,
            token_usage=TokenUsage(1, 1, 2),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


async def _call(connector: BaseConnector):
    from app.connectors.base import ConnectorConfig

    return await connector.query("prompt", "sub-query", ConnectorConfig())


def test_span_is_noop_when_disabled():
    # Default settings: no provider installed, span() yields None.
    assert not isinstance(otel_trace.get_tracer_provider(), TracerProvider)
    with span("never.recorded", {"k": "v"}) as current:
        assert current is None


@pytest.mark.asyncio
async def test_enabled_registry_wraps_connector_and_exports_spans(monkeypatch):
    exporter = ListSpanExporter()
    monkeypatch.setattr(settings, "tracing_enabled", True)
    # configure_tracing installs the provider once per process (_configured
    # guard), so this file uses a single exporter for all its assertions.
    configure_tracing(exporter_factory=lambda: exporter)
    assert isinstance(otel_trace.get_tracer_provider(), TracerProvider)

    registry = ConnectorRegistry()
    registry.register(StubConnector())
    connector = registry.get("stub")
    assert connector is not None

    response = await _call(connector)
    assert response.status == ConnectorStatus.SUCCESS

    with span("test.manual") as current:
        assert current is not None
        current.set_attribute("test.key", "value")

    provider = otel_trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    assert provider.force_flush(timeout_millis=3000)

    names = [span.name for span in exporter.spans]
    assert "connector.stub.query" in names
    assert "test.manual" in names
    target = next(span for span in exporter.spans if span.name == "connector.stub.query")
    assert target.attributes["argus.status"] == "success"
    assert target.attributes["argus.subquery_length"] == len("sub-query")


def test_disabled_registry_leaves_connectors_unwrapped():
    registry = ConnectorRegistry()
    stub = StubConnector()
    registry.register(stub)
    # Disabled tracing registers the instance untouched (same object).
    assert registry.get("stub") is stub
