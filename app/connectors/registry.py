from typing import Any

from app.config import settings
from app.connectors.base import BaseConnector
from app.tracing import span
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _wrap_with_tracing(connector: BaseConnector) -> None:
    """Attach span-emitting wrappers around query/stream_query.

    Applied once at registration time when tracing is enabled, so every
    provider call (current and future connectors) produces a span without
    any per-connector code.
    """

    async def traced_query(prompt, sub_query, config):
        with span(
            f"connector.{connector.connector_id}.query",
            {"argus.subquery_length": len(sub_query or "")},
        ) as current:
            response = await original_query(prompt, sub_query, config)
            if current is not None:
                current.set_attribute("argus.status", response.status.value)
                current.set_attribute("argus.latency_ms", response.latency_ms)
            return response

    original_query = connector.query
    connector.query = traced_query  # type: ignore[method-assign]

    if hasattr(connector, "stream_query"):

        async def traced_stream(*args: Any, **kwargs: Any):
            with span(f"connector.{connector.connector_id}.stream"):
                async for chunk in original_stream(*args, **kwargs):
                    yield chunk

        original_stream = connector.stream_query
        connector.stream_query = traced_stream  # type: ignore[method-assign]


class ConnectorRegistry:
    """Central registry for all AI model connectors.

    Connectors are registered at app startup via the lifespan handler.
    New connectors can be added by implementing BaseConnector and calling register().
    """

    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
        if settings.tracing_enabled:
            _wrap_with_tracing(connector)
        self._connectors[connector.connector_id] = connector
        logger.info({"message": "Registered connector", "connector_id": connector.connector_id})

    def get(self, connector_id: str) -> BaseConnector | None:
        return self._connectors.get(connector_id)

    def all(self) -> list[BaseConnector]:
        return list(self._connectors.values())

    def available(self) -> list[BaseConnector]:
        return [c for c in self._connectors.values() if c.is_available]

    def ids(self) -> list[str]:
        return list(self._connectors.keys())

    def __contains__(self, connector_id: str) -> bool:
        return connector_id in self._connectors


# Singleton — imported and used throughout the app
registry = ConnectorRegistry()
