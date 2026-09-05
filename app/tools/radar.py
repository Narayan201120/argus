"""RADAR paper discovery tools (Phase 4 P4-1, DEC-053).

Backend only. Real HTTP via httpx; no other network libraries.
All failures are returned as data (ToolResult ok=False), never raised.
"""

import time
from typing import Any

import httpx

from app.config import settings
from app.tools.base import BaseTool, EvidencePayload, ToolResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

RADAR_LIMIT = 10

_TITLE_MAX_CHARS = 500
_ABSTRACT_MAX_CHARS = 4000


def _clamp_confidence(value: Any, fallback: float) -> float:
    """Clamp a candidate confidence into 0..1, else return the fallback."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return fallback
    if confidence != confidence:  # NaN
        return fallback
    return max(0.0, min(1.0, confidence))


def _envelope_items(payload: Any) -> list[Any]:
    """Accept a bare list body or a {"results"|"papers"|"items": [...]} envelope."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "papers", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                return items
    return []


def _map_paper(paper: Any, rank: int, tool_name: str, latency_ms: int) -> EvidencePayload | None:
    if not isinstance(paper, dict):
        return None
    title = str(paper.get("title") or "")[:_TITLE_MAX_CHARS].strip()
    abstract = str(paper.get("abstract") or "")[:_ABSTRACT_MAX_CHARS].strip()
    if not title and not abstract:
        return None
    content = f"{title}\n{abstract}".strip()
    doi = str(paper.get("doi") or "").strip()
    if doi.startswith("http"):
        source_ref = doi
    elif "/" in doi:
        source_ref = f"https://doi.org/{doi.lstrip('/')}"
    else:
        source_ref = f"radar:{paper.get('id', rank)}"
    score = paper.get("score", paper.get("similarity_score"))
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        confidence = _clamp_confidence(score, 1.0 / (rank + 1))
    else:
        confidence = 1.0 / (rank + 1)
    return EvidencePayload(
        source_ref=source_ref,
        content=content,
        type="radar_paper",
        confidence=confidence,
        provenance={
            "tool": tool_name,
            "latency_ms": latency_ms,
            "retrieved_at": time.time(),
        },
    )


async def _get_papers(url: str, params: dict[str, Any] | None, tool_name: str) -> ToolResult:
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
            response = await client.get(url, params=params)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if response.status_code != 200:
            # Surface the upstream status (429 passthrough style) instead of masking it.
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"radar HTTP {response.status_code}: {response.text[:200]}",
                latency_ms=latency_ms,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"radar returned invalid JSON: {exc}",
                latency_ms=latency_ms,
            )
        items = [
            item
            for rank, paper in enumerate(_envelope_items(payload))
            if (item := _map_paper(paper, rank, tool_name, latency_ms)) is not None
        ]
        return ToolResult(tool_name=tool_name, ok=True, items=items, latency_ms=latency_ms)
    except httpx.HTTPError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.warning({"message": "radar request failed", "tool": tool_name, "error": str(exc)})
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error=f"radar request failed: {exc}",
            latency_ms=latency_ms,
        )
    except Exception as exc:  # never raise from run()
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.warning({"message": "radar tool error", "tool": tool_name, "error": str(exc)})
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            error=f"radar tool error: {exc}",
            latency_ms=latency_ms,
        )


class RadarSearchTool(BaseTool):
    """Keyword search over the RADAR paper index."""

    name: str = "radar_search"

    @property
    def enabled(self) -> bool:
        return settings.radar_integration_enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del params
        url = f"{settings.radar_base_url}/papers"
        return await _get_papers(url, {"q": query, "page_size": RADAR_LIMIT}, self.name)


class RadarSimilarTool(BaseTool):
    """Find papers similar to a known RADAR paper id."""

    name: str = "radar_similar"

    @property
    def enabled(self) -> bool:
        return settings.radar_integration_enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del query
        paper_id = (params or {}).get("paper_id")
        if not paper_id:
            return ToolResult(tool_name=self.name, ok=False, error="paper_id required")
        url = f"{settings.radar_base_url}/papers/{paper_id}/similar"
        return await _get_papers(url, None, self.name)


radar_search_tool = RadarSearchTool()
radar_similar_tool = RadarSimilarTool()
