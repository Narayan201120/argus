"""Research Radar workspace proxy (Phase 4 P4-4).

Read-only passthrough to the live Radar API (see docs/workspace-sources.md).
Upstream JSON bodies are returned unchanged; the frontend knows Radar shapes.
All failures map to 404/429/502; nothing raw is ever raised.
"""

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from app.config import settings
from app.metrics import PROXY_CALLS
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DISABLED_DETAIL = "Radar workspace is disabled."
_PAGE_SIZE_CAP = 50


def _api_key_headers() -> dict[str, str]:
    """Attach the Radar X-API-Key only when one is configured."""
    if settings.radar_api_key:
        return {"X-API-Key": settings.radar_api_key}
    return {}


def _check_enabled() -> None:
    """Reject workspace calls while the Radar flag is off."""
    if not settings.workspace_radar_enabled:
        PROXY_CALLS.labels(service="radar", status="disabled").inc()
        raise HTTPException(status_code=404, detail=_DISABLED_DETAIL)


async def _forward(path: str, params: dict[str, Any] | None = None) -> Response:
    """GET one Radar path and return the upstream JSON body unchanged."""
    url = f"{settings.radar_base_url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
            upstream = await client.get(url, params=params, headers=_api_key_headers())
    except httpx.TimeoutException as exc:
        PROXY_CALLS.labels(service="radar", status="timeout").inc()
        logger.warning({"message": "radar proxy timed out", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="Radar request timed out.") from exc
    except httpx.HTTPError as exc:
        PROXY_CALLS.labels(service="radar", status="error").inc()
        logger.warning({"message": "radar proxy request failed", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="Radar request failed.") from exc
    except Exception as exc:  # never let anything raw escape the proxy
        PROXY_CALLS.labels(service="radar", status="error").inc()
        logger.warning({"message": "radar proxy error", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="Radar request failed.") from exc
    if upstream.status_code == 429:
        PROXY_CALLS.labels(service="radar", status="error").inc()
        retry_after = upstream.headers.get("retry-after")
        headers = {"Retry-After": retry_after} if retry_after is not None else None
        raise HTTPException(
            status_code=429, detail="Radar rate limit exceeded.", headers=headers
        )
    if upstream.status_code != 200:
        PROXY_CALLS.labels(service="radar", status="error").inc()
        logger.warning(
            {"message": "radar proxy upstream error", "url": url, "status": upstream.status_code}
        )
        raise HTTPException(
            status_code=502, detail=f"Radar HTTP {upstream.status_code}."
        )
    try:
        payload: Any = upstream.json()
    except ValueError as exc:
        PROXY_CALLS.labels(service="radar", status="error").inc()
        raise HTTPException(status_code=502, detail="Radar returned invalid JSON.") from exc
    PROXY_CALLS.labels(service="radar", status="ok").inc()
    return JSONResponse(content=payload)


@router.get("/radar/papers")
async def list_papers(
    q: str | None = Query(default=None),
    year: int | None = Query(default=None),
    topic: str | None = Query(default=None),
    author: str | None = Query(default=None),
    page: int = Query(default=1),
    page_size: int = Query(default=20),
) -> Response:
    """Proxy GET /papers; page_size capped at 50 for UI sanity."""
    _check_enabled()
    params: dict[str, Any] = {"page": page, "page_size": min(page_size, _PAGE_SIZE_CAP)}
    if q is not None:
        params["q"] = q
    if year is not None:
        params["year"] = year
    if topic is not None:
        params["topic"] = topic
    if author is not None:
        params["author"] = author
    return await _forward("/papers", params)


@router.get("/radar/papers/{paper_id}")
async def get_paper(paper_id: str) -> Response:
    """Proxy GET /papers/{id} with no query params."""
    _check_enabled()
    return await _forward(f"/papers/{paper_id}")


@router.get("/radar/papers/{paper_id}/similar")
async def get_similar_papers(paper_id: str) -> Response:
    """Proxy GET /papers/{id}/similar with no query params."""
    _check_enabled()
    return await _forward(f"/papers/{paper_id}/similar")
