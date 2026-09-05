"""P4-1 tool registry (DEC-053, backend only, mock-only)."""

from app.tools.base import BaseTool
from app.tools.radar import radar_search_tool, radar_similar_tool
from app.tools.rag import rag_retrieve_tool
from app.tools.stubs import web_fetch_tool, web_search_tool

ALL_TOOLS: list[BaseTool] = [
    radar_search_tool,
    radar_similar_tool,
    rag_retrieve_tool,
    web_search_tool,
    web_fetch_tool,
]


def build_tool_registry() -> dict[str, BaseTool]:
    """Build a fresh {tool.name: tool} dict on every call.

    Never cache at import; tests monkeypatch settings flags.

    Dispatcher convention (for the sibling worker): iterate ALL tools,
    skip disabled ones WITHOUT consuming budget, record metric status "disabled".
    """
    registry: dict[str, BaseTool] = {}
    for tool in ALL_TOOLS:
        if tool.name in registry:
            raise ValueError(f"duplicate tool name: {tool.name}")
        registry[tool.name] = tool
    return registry
