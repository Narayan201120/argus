"""Dead-provider demotion (P4-5d, backend only, mock-only).

A connector is demoted after 3 consecutive AUTH failures (bad/revoked keys).
Any success resets the counter. Quota / rate-limit / timeout / 5xx signals
must NEVER count toward demotion.

State is an in-memory dict keyed by connector_id: process restarts reset all
counters (no persistence by design; mock-only scope).
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)

DEMOTION_THRESHOLD = 3

_counts: dict[str, int] = {}

# Substrings (lowercased) identifying AUTH-class failures: bad/revoked keys.
_AUTH_MARKERS: tuple[str, ...] = (
    "401",
    "403",
    "unauthorized",
    "invalid_api_key",
    "invalid key",
    "invalid-key",
    "invalid api",
    "incorrect api",
    "bad api key",
    "wrong api key",
    "authentication failed",
    "authentication error",
    "invalid authentication",
    "forbidden",
    "api key not valid",
    "api key required",
    "missing api key",
    "no api key",
)

# Substrings (lowercased) that must NEVER demote, even alongside auth words.
_EXCLUSION_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "quota",
    "resource_exhausted",
    "timeout",
    "timed out",
    "500",
    "502",
    "503",
    "504",
)


def is_auth_failure(status: str, error: str | None) -> bool:
    """Classify a connector failure as AUTH-class (bad/revoked key).

    Args:
        status: Connector status value (e.g. "success", "error",
            "rate_limited", "timeout") or an HTTP-code-like string.
        error: Provider error text, if any.

    Returns True only for 401/403/unauthorized/invalid-key class signals.
    Rate limits, timeouts, 5xx, and quota signals always return False.
    """
    normalized_status = str(status or "").lower()
    if normalized_status in ("success", "rate_limited", "timeout", "skipped"):
        return False
    err = (error or "").lower()
    for marker in _EXCLUSION_MARKERS:
        if marker in err:
            return False
    if "not configured" in err and ("key" in err or "api" in err):
        return True
    combined = f"{normalized_status} {err}"
    return any(marker in combined for marker in _AUTH_MARKERS)


def record_auth_failure(connector_id: str) -> int:
    """Record one AUTH failure; return the new consecutive count."""
    count = _counts.get(connector_id, 0) + 1
    _counts[connector_id] = count
    if count == DEMOTION_THRESHOLD:
        logger.warning(
            {
                "message": "Connector demoted after consecutive AUTH failures",
                "connector_id": connector_id,
                "consecutive_auth_failures": count,
            }
        )
    return count


def record_success(connector_id: str) -> None:
    """Reset the AUTH-failure counter; log restore transitions."""
    previous = _counts.get(connector_id, 0)
    if not previous:
        return
    _counts[connector_id] = 0
    if previous >= DEMOTION_THRESHOLD:
        logger.warning(
            {
                "message": "Connector restored after success",
                "connector_id": connector_id,
                "previous_failures": previous,
            }
        )


def consecutive_auth_failures(connector_id: str) -> int:
    """Current consecutive AUTH-failure count (0 when unknown)."""
    return _counts.get(connector_id, 0)


def is_demoted(connector_id: str) -> bool:
    """True once consecutive AUTH failures reach the threshold."""
    return _counts.get(connector_id, 0) >= DEMOTION_THRESHOLD


def reset_all() -> None:
    """Clear all counters (tests / restarts)."""
    _counts.clear()
