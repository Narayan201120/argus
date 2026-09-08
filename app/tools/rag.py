"""RAG retrieval tool (Phase 4 P4-1, DEC-053).

Temporary D7 service-identity model: the backend authenticates to the RAG
service with a shared service username/password, caches the bearer token in
memory, and refreshes it on expiry or on a single 401 retry. Backend only.
Real HTTP via httpx; no other network libraries.

P7-2 per-user exchange (Option C, DEC-056): in multi-user mode the backend
mints a short-lived HS256 assertion per subject and exchanges it for a
per-subject bearer token. Per-subject tokens are cached in _user_tokens.
Multi-user mode without exchange configured raises instead of falling back
to the shared service identity. No tokens or secrets are ever logged.
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import jwt

from app.config import settings
from app.tools.base import BaseTool, EvidencePayload, ToolResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

RAG_TOP_K = 10
RAG_EXCHANGE_DEFAULT_PATH = "/api/service/exchange/"
_ASSERTION_TTL_S = 60
_CHUNK_MAX_CHARS = 4000
_TOKEN_SKEW_S = 30.0
_DEFAULT_TOKEN_TTL_S = 1500.0  # RAG access lifetime is 30 min; refresh early


def _confidence_or(value: Any, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return fallback
    if confidence != confidence:  # NaN
        return fallback
    return max(0.0, min(1.0, confidence))


def _envelope_items(payload: Any) -> list[Any]:
    """Accept a bare list body or a {"results"|"chunks"|"items": [...]} envelope."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "chunks", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                return items
    return []


def mint_exchange_assertion(subject: str) -> str:
    """Mint a short-lived HS256 assertion for the per-user exchange."""
    secret = settings.rag_exchange_secret
    if not secret:
        raise RuntimeError("rag exchange secret is not configured")
    now = datetime.now(UTC)
    iat = int(now.timestamp())
    payload: dict[str, Any] = {
        "sub": subject,
        "iss": settings.jwt_issuer or "argus",
        "iat": iat,
        "exp": iat + _ASSERTION_TTL_S,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def exchange_url() -> str:
    """Exchange endpoint: explicit override wins, else base_url + default path."""
    if settings.rag_exchange_url:
        return settings.rag_exchange_url
    return (settings.rag_base_url or "").rstrip("/") + RAG_EXCHANGE_DEFAULT_PATH


def _exchange_configured() -> bool:
    return bool(
        settings.rag_exchange_enabled
        and settings.rag_exchange_secret
        and settings.rag_base_url
    )


class RagRetrieveTool(BaseTool):
    """Semantic retrieval over the RAG service chunk index."""

    name: str = "rag_retrieve"

    def __init__(self) -> None:
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._user_tokens: dict[str, tuple[str, float]] = {}
        # Serializes sign-in refreshes so a herd of concurrent callers
        # (retrieval + library proxy share this singleton) fires one POST.
        self._token_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        if not (settings.rag_integration_enabled and settings.rag_base_url):
            return False
        if settings.rag_service_user and settings.rag_service_pass:
            return True
        return bool(settings.rag_exchange_enabled and settings.rag_exchange_secret)

    def _invalidate_subject(self, subject: str) -> None:
        self._user_tokens.pop(subject, None)
        if subject == "local":
            self._access_token = None
            self._token_expires_at = 0.0

    async def _exchange_token(self, subject: str) -> str:
        url = exchange_url()
        try:
            async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
                response = await client.post(
                    url,
                    json={"assertion": mint_exchange_assertion(subject)},
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"rag exchange failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(f"rag exchange HTTP {response.status_code}: {response.text[:200]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"rag exchange returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("rag exchange returned an unexpected body")
        tokens = payload.get("tokens")
        if isinstance(tokens, dict):
            token = tokens.get("access")
        else:
            token = payload.get("access") or payload.get("access_token") or payload.get("token")
        if not token:
            raise RuntimeError("rag exchange returned no access token")
        ttl: Any = payload.get("expires_in", _DEFAULT_TOKEN_TTL_S)
        try:
            ttl_seconds = float(ttl)
        except (TypeError, ValueError):
            ttl_seconds = _DEFAULT_TOKEN_TTL_S
        if ttl_seconds <= 0:
            ttl_seconds = _DEFAULT_TOKEN_TTL_S
        token_str = str(token)
        self._user_tokens[subject] = (token_str, time.time() + ttl_seconds)
        return token_str

    async def _ensure_token(self, sub: str = "local") -> str:
        """Return a cached bearer token or sign in for a fresh one."""
        if _exchange_configured():
            entry = self._user_tokens.get(sub)
            if entry and time.time() < entry[1] - _TOKEN_SKEW_S:
                return entry[0]
            async with self._token_lock:
                entry = self._user_tokens.get(sub)
                if entry and time.time() < entry[1] - _TOKEN_SKEW_S:
                    return entry[0]
                return await self._exchange_token(sub)
        if not settings.auth_single_user_mode:
            raise RuntimeError("per-user RAG exchange is not configured")
        if self._access_token and time.time() < self._token_expires_at - _TOKEN_SKEW_S:
            return self._access_token
        async with self._token_lock:
            if self._access_token and time.time() < self._token_expires_at - _TOKEN_SKEW_S:
                return self._access_token
            base_url = settings.rag_base_url or ""
            try:
                async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
                    response = await client.post(
                        f"{base_url}/api/sign-in/",
                        json={"username": settings.rag_service_user, "password": settings.rag_service_pass},
                    )
            except httpx.HTTPError as exc:
                raise RuntimeError(f"rag sign-in failed: {exc}") from exc
            if response.status_code != 200:
                raise RuntimeError(f"rag sign-in HTTP {response.status_code}: {response.text[:200]}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError(f"rag sign-in returned invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("rag sign-in returned an unexpected body")
            tokens = payload.get("tokens")
            if isinstance(tokens, dict):
                token = tokens.get("access")
            else:
                token = payload.get("access") or payload.get("access_token") or payload.get("token")
            if not token:
                raise RuntimeError("rag sign-in returned no access token")
            ttl: Any = payload.get("expires_in", _DEFAULT_TOKEN_TTL_S)
            try:
                ttl_seconds = float(ttl)
            except (TypeError, ValueError):
                ttl_seconds = _DEFAULT_TOKEN_TTL_S
            if ttl_seconds <= 0:
                ttl_seconds = _DEFAULT_TOKEN_TTL_S
            self._access_token = str(token)
            self._token_expires_at = time.time() + ttl_seconds
            return self._access_token

    def _map_hit(self, hit: Any, index: int, latency_ms: int) -> EvidencePayload | None:
        if not isinstance(hit, dict):
            return None
        content = hit.get("chunk")
        if content is None:
            content = hit.get("text", hit.get("content"))
        content = str(content or "")[:_CHUNK_MAX_CHARS].strip()
        if not content:
            return None
        source_ref = (
            hit.get("source") or hit.get("doc") or hit.get("reference") or hit.get("url") or hit.get("id")
        )
        return EvidencePayload(
            source_ref=str(source_ref) if source_ref else f"rag:{index}",
            content=content,
            type="rag_chunk",
            confidence=_confidence_or(hit.get("score"), 0.5),
            provenance={
                "tool": self.name,
                "latency_ms": latency_ms,
                "retrieved_at": time.time(),
            },
        )

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        owner = ((params or {}).get("owner")) or "local"
        start = time.perf_counter()
        base_url = settings.rag_base_url
        if not base_url:
            return ToolResult(tool_name=self.name, ok=False, error="rag_base_url is not configured")
        try:
            token = await self._ensure_token(owner)
            headers = {"Authorization": f"Bearer {token}"}
            try:
                async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
                    response = await client.post(
                        f"{base_url}/api/search/",
                        json={"query": query, "top_k": RAG_TOP_K},
                        headers=headers,
                    )
                    if response.status_code == 401:
                        # Token rejected: refresh once and retry once.
                        self._invalidate_subject(owner)
                        token = await self._ensure_token(owner)
                        headers = {"Authorization": f"Bearer {token}"}
                        response = await client.post(
                            f"{base_url}/api/search/",
                            json={"query": query, "top_k": RAG_TOP_K},
                            headers=headers,
                        )
            except httpx.HTTPError as exc:
                latency_ms = int((time.perf_counter() - start) * 1000)
                logger.warning({"message": "rag request failed", "tool": self.name, "error": str(exc)})
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"rag request failed: {exc}",
                    latency_ms=latency_ms,
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            if response.status_code != 200:
                # Surface the upstream status (429 passthrough style) instead of masking it.
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"rag HTTP {response.status_code}: {response.text[:200]}",
                    latency_ms=latency_ms,
                )
            try:
                payload = response.json()
            except ValueError as exc:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"rag returned invalid JSON: {exc}",
                    latency_ms=latency_ms,
                )
            items = [
                item
                for index, hit in enumerate(_envelope_items(payload))
                if (item := self._map_hit(hit, index, latency_ms)) is not None
            ]
            return ToolResult(tool_name=self.name, ok=True, items=items, latency_ms=latency_ms)
        except RuntimeError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return ToolResult(tool_name=self.name, ok=False, error=str(exc), latency_ms=latency_ms)
        except Exception as exc:  # never raise from run()
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning({"message": "rag tool error", "tool": self.name, "error": str(exc)})
            return ToolResult(
                tool_name=self.name, ok=False, error=f"rag tool error: {exc}", latency_ms=latency_ms
            )


rag_retrieve_tool = RagRetrieveTool()
