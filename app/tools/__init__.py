"""P4-1 tool abstraction + registry re-exports (DEC-053, backend only, mock-only)."""

from app.tools.base import BaseTool, EvidencePayload, ToolResult
from app.tools.registry import build_tool_registry

__all__ = ["BaseTool", "EvidencePayload", "ToolResult", "build_tool_registry"]
