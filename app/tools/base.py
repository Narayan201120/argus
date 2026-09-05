"""P4-1 tool abstraction (DEC-053, backend only, mock-only)."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class EvidencePayload(BaseModel):
    # Dispatcher stamps id/investigation_id/created_at; tools never assign them.
    source_ref: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1, max_length=20000)
    type: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    ok: bool
    items: list[EvidencePayload] = Field(default_factory=list)
    error: str | None = None
    latency_ms: int = Field(default=0, ge=0)


class BaseTool(ABC):
    name: str

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this tool may run; each subclass reads its own flag from settings."""
        raise NotImplementedError

    @abstractmethod
    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        """Run the tool; concrete tools measure their own latency_ms."""
        raise NotImplementedError
