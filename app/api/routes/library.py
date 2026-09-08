"""Document library proxy (Phase 4 P4-4).

Read-only passthrough to the live RAG API (see docs/workspace-sources.md).
Upstream JSON bodies are returned unchanged; the frontend knows RAG shapes.
All failures map to 404/429/502; nothing raw is ever raised.

Auth sharing: the service-identity sign-in lives on the rag_retrieve_tool
singleton in app/tools/rag.py. This proxy reuses its cached token method
instead of duplicating sign-in logic.
"""

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.auth import resolve_subject
from app.config import settings
from app.metrics import PROXY_CALLS
from app.tools.rag import rag_retrieve_tool
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DISABLED_DETAIL = "Document library is disabled."


def _check_enabled() -> None:
    """Reject library calls while the RAG flag is off."""
    if not settings.workspace_rag_enabled:
        PROXY_CALLS.labels(service="rag", status="disabled").inc()
        raise HTTPException(status_code=404, detail=_DISABLED_DETAIL)


async def _service_headers(owner: str = "local") -> dict[str, str]:
    """Build the Bearer headers from the shared tool-singleton token.

    Per-user (P7-2): the singleton signs in per subject. Duck-typed for
    backward compatibility: prefer _ensure_token(owner) when it accepts a
    subject, else fall back to the legacy zero-arg _ensure_token().
    """
    ensure = getattr(rag_retrieve_tool, "_ensure_token")  # noqa: B009 - intentional duck-typing for backward compat
    try:
        # Positional subject; raises TypeError on the legacy zero-arg form.
        token = await ensure(owner)
    except TypeError:
        token = await ensure()
    return {"Authorization": f"Bearer {token}"}


def _invalidate_token(owner: str = "local") -> None:
    """Clear one subject's cached token so the next call signs in fresh.

    Duck-typed: prefer _invalidate_subject(owner) when present, else fall
    back to the singleton's legacy _invalidate_token() or direct attr
    clearing for the single-token shape.
    """
    invalidate_subject = getattr(rag_retrieve_tool, "_invalidate_subject", None)
    if callable(invalidate_subject):
        invalidate_subject(owner)
        return
    legacy_invalidate = getattr(rag_retrieve_tool, "_invalidate_token", None)
    if callable(legacy_invalidate):
        try:
            legacy_invalidate(owner)
        except TypeError:
            legacy_invalidate()
        return
    rag_retrieve_tool._access_token = None
    rag_retrieve_tool._token_expires_at = 0.0


async def _forward(path: str, owner: str = "local") -> Response:
    """GET one RAG path and return the upstream JSON body unchanged."""
    base_url = (settings.rag_base_url or "").rstrip("/")
    if not base_url:
        PROXY_CALLS.labels(service="rag", status="error").inc()
        raise HTTPException(status_code=502, detail="RAG library is not configured.")
    url = f"{base_url}{path}"
    try:
        headers = await _service_headers(owner)
    except Exception as exc:  # sign-in failure maps to 502, never raw
        PROXY_CALLS.labels(service="rag", status="error").inc()
        logger.warning({"message": "rag proxy sign-in failed", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="RAG sign-in failed.") from exc
    try:
        async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
            upstream = await client.get(url, headers=headers)
            if upstream.status_code == 401:
                # Token rejected: refresh once and retry once (per-subject).
                _invalidate_token(owner)
                try:
                    headers = await _service_headers(owner)
                except Exception as exc:
                    PROXY_CALLS.labels(service="rag", status="error").inc()
                    logger.warning(
                        {"message": "rag proxy sign-in failed", "url": url, "error": str(exc)}
                    )
                    raise HTTPException(status_code=502, detail="RAG sign-in failed.") from exc
                upstream = await client.get(url, headers=headers)
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        PROXY_CALLS.labels(service="rag", status="timeout").inc()
        logger.warning({"message": "rag proxy timed out", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="RAG library request timed out.") from exc
    except httpx.HTTPError as exc:
        PROXY_CALLS.labels(service="rag", status="error").inc()
        logger.warning({"message": "rag proxy request failed", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="RAG library request failed.") from exc
    except Exception as exc:  # never let anything raw escape the proxy
        PROXY_CALLS.labels(service="rag", status="error").inc()
        logger.warning({"message": "rag proxy error", "url": url, "error": str(exc)})
        raise HTTPException(status_code=502, detail="RAG library request failed.") from exc
    if upstream.status_code == 401:
        # Still unauthorized after the single refresh-and-retry.
        PROXY_CALLS.labels(service="rag", status="error").inc()
        logger.warning({"message": "rag proxy unauthorized", "url": url, "status": 401})
        raise HTTPException(status_code=502, detail="RAG HTTP 401.")
    if upstream.status_code == 429:
        PROXY_CALLS.labels(service="rag", status="error").inc()
        retry_after = upstream.headers.get("retry-after")
        headers_out = {"Retry-After": retry_after} if retry_after is not None else None
        raise HTTPException(
            status_code=429, detail="RAG library rate limit exceeded.", headers=headers_out
        )
    if upstream.status_code != 200:
        PROXY_CALLS.labels(service="rag", status="error").inc()
        logger.warning(
            {"message": "rag proxy upstream error", "url": url, "status": upstream.status_code}
        )
        raise HTTPException(status_code=502, detail=f"RAG HTTP {upstream.status_code}.")
    try:
        payload: Any = upstream.json()
    except ValueError as exc:
        PROXY_CALLS.labels(service="rag", status="error").inc()
        raise HTTPException(status_code=502, detail="RAG returned invalid JSON.") from exc
    PROXY_CALLS.labels(service="rag", status="ok").inc()
    return JSONResponse(content=payload)


@router.get("/library/documents")
async def list_documents(http_request: Request) -> Response:
    """Proxy GET /api/documents/ with no query params."""
    _check_enabled()
    return await _forward("/api/documents/", resolve_subject(http_request))


@router.get("/library/documents/{filename}")
async def get_document(filename: str, http_request: Request) -> Response:
    """Proxy GET /api/documents/<filename>/ with no query params."""
    _check_enabled()
    return await _forward(f"/api/documents/{filename}/", resolve_subject(http_request))


@router.get("/library/collections")
async def list_collections(http_request: Request) -> Response:
    """Proxy GET /api/collections/ with no query params."""
    _check_enabled()
    return await _forward("/api/collections/", resolve_subject(http_request))


@router.get("/library/collections/{collection_id}")
async def get_collection(collection_id: str, http_request: Request) -> Response:
    """Proxy GET /api/collections/<id>/ with no query params."""
    _check_enabled()
    return await _forward(f"/api/collections/{collection_id}/", resolve_subject(http_request))
