from app.connectors.base import BaseConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectorRegistry:
    """Central registry for all AI model connectors.

    Connectors are registered at app startup via the lifespan handler.
    New connectors can be added by implementing BaseConnector and calling register().
    """

    def __init__(self):
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
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
