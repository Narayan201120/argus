from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum


class ConnectorStatus(StrEnum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ConnectorResponse:
    model_id: str
    content: str
    latency_ms: int
    token_usage: TokenUsage
    status: ConnectorStatus
    error: str | None = None
    sub_query: str | None = None  # The sub-query this connector was assigned


@dataclass
class ConnectorConfig:
    timeout_s: int = 45
    max_retries: int = 1
    temperature: float = 0.7
    max_tokens: int = 4096
    model_override: str | None = None  # Override the connector's default model


class BaseConnector(ABC):
    """Abstract base class for all AI model connectors.

    Every connector must implement:
    - query(): Send a sub-query to the model and return a ConnectorResponse
    - health_check(): Verify the upstream API is reachable

    Connectors register themselves in the ConnectorRegistry at startup.
    """

    connector_id: str       # Unique ID e.g. "gemini", "openai", "claude"
    display_name: str       # Human-readable name
    capabilities: list[str] # e.g. ["text", "vision", "code", "research"]
    is_available: bool = True

    @abstractmethod
    async def query(
        self,
        prompt: str,
        sub_query: str,
        config: ConnectorConfig,
    ) -> ConnectorResponse:
        """Send a query to the model.

        Args:
            prompt: The original user query (context only)
            sub_query: The decomposed sub-query this connector should answer
            config: Per-call configuration (timeout, tokens, temperature)
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the connector's API key is set and upstream is reachable."""
        ...

    def capability_profile(self) -> dict:
        return {
            "connector_id": self.connector_id,
            "display_name": self.display_name,
            "capabilities": self.capabilities,
            "is_available": self.is_available,
        }

    async def stream_query(
        self,
        prompt: str,
        sub_query: str,
        config: ConnectorConfig,
    ) -> AsyncIterator[str]:
        """Incremental text output for streaming endpoints.

        Default implementation delegates to query() and yields the full
        content once. Providers with native streaming APIs override this
        to yield real deltas.
        """
        response = await self.query(prompt, sub_query, config)
        if response.status != ConnectorStatus.SUCCESS or not response.content:
            return
        yield response.content
