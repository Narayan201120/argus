"""Real web search + fetch tools (Phase 4 P4-5a, DEC-053).

Backend only. Tavily search POSTs to the Tavily API; fetch downloads a page
and extracts readable text with trafilatura. All failures are returned as
data (ToolResult ok=False), never raised. Tests must mock httpx/trafilatura;
no live calls.
"""

import asyncio
import ipaddress
import socket
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

from app.config import settings
from app.tools.base import BaseTool, EvidencePayload, ToolResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5

ANSWER_MAX_CHARS = 4000
RESULT_MAX_CHARS = 4000
PAGE_MAX_CHARS = 8000
MAX_FETCH_BYTES = 2 * 1024 * 1024

ANSWER_CONFIDENCE = 0.7
PAGE_CONFIDENCE = 0.6


def _clamp_confidence(value: Any, fallback: float) -> float:
    """Clamp a candidate confidence into 0..1, else return the fallback."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return fallback
    if confidence != confidence:  # NaN
        return fallback
    return max(0.0, min(1.0, confidence))


def _search_items(payload: Any) -> list[Any]:
    """Accept a bare-list body or a dict with results under results/items."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                return items
    return []


def _map_search_result(
    item: Any, rank: int, tool_name: str, latency_ms: int
) -> EvidencePayload | None:
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or "").strip()
    if not url:
        return None
    title = str(item.get("title") or "").strip()
    snippet = str(item.get("content") or "").strip()
    content = f"{title}\n{snippet}".strip()[:RESULT_MAX_CHARS].strip()
    if not content:
        return None
    try:
        return EvidencePayload(
            source_ref=url,
            content=content,
            type="web_result",
            confidence=_clamp_confidence(item.get("score"), 1.0 / (rank + 1)),
            provenance={
                "tool": tool_name,
                "latency_ms": latency_ms,
                "retrieved_at": time.time(),
            },
        )
    except Exception:
        return None


def _ssrf_block_reason(url: str) -> str | None:
    """Return why a fetch URL is refused, or None when it looks fetchable.

    Fast path only: refuses non-http(s) schemes and localhost/loopback/private
    literal IPs (via ipaddress). Hostnames are NOT resolved here; use
    _dns_block_reason for that. Redirect targets are NOT checked here; the
    fetch path re-checks the final response URL separately.
    """
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return f"invalid url: {exc}"
    if parsed.scheme not in ("http", "https"):
        return f"refused non-http(s) url scheme: {parsed.scheme or '(missing)'}"
    host = parsed.hostname
    if not host:
        return "url has no host"
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return "refused localhost host"
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return None
    if _ip_is_blocked(ip):
        return f"refused loopback/private host: {host}"
    return None


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True when a resolved/literal IP must never be fetched."""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def _hostname_of(url: str) -> str | None:
    """Extract the hostname from a URL string, or None when absent/invalid."""
    try:
        return urlparse(str(url)).hostname
    except ValueError:
        return None


async def _dns_block_reason(host: str) -> str | None:
    """Resolve a hostname and refuse it if ANY answer is non-public.

    Literal IPs need no DNS (the _ssrf_block_reason fast path already vetted
    them) and return None here. DNS failures, empty answers, or unparseable
    answers refuse fail-closed since safety cannot be verified.

    Limits (honest, not fixed here): TOCTOU remains — DNS may change between
    this check and the httpx connect, and per-hop redirect targets are not
    pinned; only the initial URL (before connect) and the final response URL
    (after redirects) are checked. A malicious DNS/redirect can still race
    the check. Mitigating that needs connect-level IP pinning, which httpx
    does not offer here.
    """
    if not host:
        return "url has no host"
    normalized = host.lower().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return "refused localhost host"
    try:
        ipaddress.ip_address(normalized)
        return None
    except ValueError:
        pass
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, normalized, None)
    except Exception as exc:
        return f"refused unverifiable host (dns failed): {host}: {exc}"
    if not infos:
        return f"refused unverifiable host (no dns results): {host}"
    for info in infos:
        try:
            sockaddr = info[4]
            if isinstance(sockaddr, (tuple, list)):
                ip_raw = sockaddr[0]
            else:
                ip_raw = sockaddr
            ip = ipaddress.ip_address(str(ip_raw))
        except ValueError:
            return f"refused unverifiable host resolution: {host}"
        if _ip_is_blocked(ip):
            return f"refused loopback/private host: {host}"
    return None


class TavilySearchTool(BaseTool):
    """Tavily web search; the synthesised answer (when present) comes first."""

    name: str = "web_search"

    @property
    def enabled(self) -> bool:
        return settings.web_tools_enabled and bool(settings.tavily_api_key)

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        del params
        start = time.perf_counter()
        api_key = settings.tavily_api_key
        if not api_key:
            return ToolResult(
                tool_name=self.name, ok=False, error="tavily_api_key is not configured"
            )
        try:
            async with httpx.AsyncClient(timeout=settings.tool_timeout_s) as client:
                response = await client.post(
                    TAVILY_SEARCH_URL,
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": TAVILY_MAX_RESULTS,
                        "include_answer": True,
                        "search_depth": "basic",
                    },
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            if response.status_code != 200:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"tavily HTTP {response.status_code}: {response.text[:200]}",
                    latency_ms=latency_ms,
                )
            try:
                payload = response.json()
            except ValueError as exc:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"tavily returned invalid JSON: {exc}",
                    latency_ms=latency_ms,
                )
            items: list[EvidencePayload] = []
            if isinstance(payload, dict):
                answer = str(payload.get("answer") or "").strip()[:ANSWER_MAX_CHARS].strip()
                if answer:
                    items.append(
                        EvidencePayload(
                            source_ref="tavily:answer",
                            content=answer,
                            type="web_answer",
                            confidence=ANSWER_CONFIDENCE,
                            provenance={"tool": self.name, "latency_ms": latency_ms},
                        )
                    )
            for rank, raw in enumerate(_search_items(payload)):
                mapped = _map_search_result(raw, rank, self.name, latency_ms)
                if mapped is not None:
                    items.append(mapped)
            return ToolResult(tool_name=self.name, ok=True, items=items, latency_ms=latency_ms)
        except httpx.HTTPError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning({"message": "tavily request failed", "tool": self.name, "error": str(exc)})
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"tavily request failed: {exc}",
                latency_ms=latency_ms,
            )
        except Exception as exc:  # never raise from run()
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning({"message": "tavily tool error", "tool": self.name, "error": str(exc)})
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"tavily tool error: {exc}",
                latency_ms=latency_ms,
            )


class WebFetchTool(BaseTool):
    """Fetch a page URL and extract readable text with trafilatura.

    The URL comes from params["url"]; a query starting with http(s):// is
    also accepted. Gap queries are usually topics, so callers pass the URL
    via params.

    SSRF guard: literal-IP/scheme fast path (_ssrf_block_reason) plus DNS
    resolution (_dns_block_reason, fail-closed) before connect, then the
    final response URL is re-checked the same way after redirects. TOCTOU
    remains: DNS may change between check and connect and intermediate
    redirect hops are not individually pinned (only the final URL).
    """

    name: str = "web_fetch"

    @property
    def enabled(self) -> bool:
        return settings.web_tools_enabled

    async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
        start = time.perf_counter()
        candidate = (params or {}).get("url")
        url = candidate.strip() if isinstance(candidate, str) else ""
        if not url:
            stripped = query.strip()
            lowered = stripped.lower()
            if lowered.startswith("http://") or lowered.startswith("https://"):
                url = stripped
        if not url:
            return ToolResult(tool_name=self.name, ok=False, error="url required")
        blocked = _ssrf_block_reason(url)
        if blocked is not None:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=blocked,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        initial_host = _hostname_of(url)
        if initial_host is None:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error="url has no host",
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        dns_blocked = await _dns_block_reason(initial_host)
        if dns_blocked is not None:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=dns_blocked,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        try:
            async with httpx.AsyncClient(
                timeout=settings.tool_timeout_s, follow_redirects=True
            ) as client:
                async with client.stream("GET", url) as response:
                    final_raw = getattr(response, "url", "")
                    final_url = str(final_raw or "")
                    if final_url:
                        final_blocked = _ssrf_block_reason(final_url)
                        if final_blocked is not None:
                            return ToolResult(
                                tool_name=self.name,
                                ok=False,
                                error=final_blocked,
                                latency_ms=int((time.perf_counter() - start) * 1000),
                            )
                        final_host = _hostname_of(final_url)
                        if final_host is None:
                            return ToolResult(
                                tool_name=self.name,
                                ok=False,
                                error="url has no host",
                                latency_ms=int((time.perf_counter() - start) * 1000),
                            )
                        if final_host.lower().rstrip(".") != initial_host.lower().rstrip("."):
                            final_dns_blocked = await _dns_block_reason(final_host)
                            if final_dns_blocked is not None:
                                return ToolResult(
                                    tool_name=self.name,
                                    ok=False,
                                    error=final_dns_blocked,
                                    latency_ms=int((time.perf_counter() - start) * 1000),
                                )
                    length = response.headers.get("content-length")
                    if length is not None:
                        try:
                            if int(length) > MAX_FETCH_BYTES:
                                return ToolResult(
                                    tool_name=self.name,
                                    ok=False,
                                    error=f"fetch refused: body larger than {MAX_FETCH_BYTES} bytes",
                                    latency_ms=int((time.perf_counter() - start) * 1000),
                                )
                        except (TypeError, ValueError):
                            pass
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        body.extend(chunk)
                        if len(body) > MAX_FETCH_BYTES:
                            del body[MAX_FETCH_BYTES:]
                            break
                    status = response.status_code
                    charset = response.charset_encoding or "utf-8"
            try:
                html = bytes(body).decode(charset, errors="replace")
            except (LookupError, ValueError):
                html = bytes(body).decode("utf-8", errors="replace")
            latency_ms = int((time.perf_counter() - start) * 1000)
            if status != 200:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"fetch HTTP {status}: {html[:200]}",
                    latency_ms=latency_ms,
                )
            try:
                extracted = await asyncio.to_thread(trafilatura.extract, html)
            except Exception as exc:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"fetch extraction failed: {exc}",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
            text = str(extracted or "").strip()[:PAGE_MAX_CHARS].strip()
            if not text:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error="fetch extracted no readable text",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                )
            latency_ms = int((time.perf_counter() - start) * 1000)
            try:
                item = EvidencePayload(
                    source_ref=url,
                    content=text,
                    type="web_page",
                    confidence=PAGE_CONFIDENCE,
                    provenance={
                        "tool": self.name,
                        "latency_ms": latency_ms,
                        "retrieved_at": time.time(),
                    },
                )
            except Exception as exc:
                return ToolResult(
                    tool_name=self.name,
                    ok=False,
                    error=f"fetch evidence invalid: {exc}",
                    latency_ms=latency_ms,
                )
            return ToolResult(tool_name=self.name, ok=True, items=[item], latency_ms=latency_ms)
        except httpx.HTTPError as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning({"message": "fetch request failed", "tool": self.name, "error": str(exc)})
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"fetch request failed: {exc}",
                latency_ms=latency_ms,
            )
        except Exception as exc:  # never raise from run()
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.warning({"message": "fetch tool error", "tool": self.name, "error": str(exc)})
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"fetch tool error: {exc}",
                latency_ms=latency_ms,
            )


web_search_tool = TavilySearchTool()
web_fetch_tool = WebFetchTool()
