"""Placeholder web tools (Phase 4 P4-1, DEC-053).

Interface-identical placeholder; P4-5 swaps the implementation for real web
search/fetch. Until then these report not-configured when disabled.
"""

from typing import Any

from app.config import settings
from app.tools.base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Web search placeholder (real implementation lands in P4-5)."""

    name: str = "web_search"

    @property
    def enabled(self) -> bool:
        return settings.web_tools_enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del query, params
        return ToolResult(
            tool_name=self.name, ok=False, error=f"{self.name} is not configured (WEB_TOOLS_ENABLED=false)"
        )


class WebFetchTool(BaseTool):
    """Web fetch placeholder (real implementation lands in P4-5)."""

    name: str = "web_fetch"

    @property
    def enabled(self) -> bool:
        return settings.web_tools_enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del query, params
        return ToolResult(
            tool_name=self.name, ok=False, error=f"{self.name} is not configured (WEB_TOOLS_ENABLED=false)"
        )


web_search_tool = WebSearchTool()
web_fetch_tool = WebFetchTool()
