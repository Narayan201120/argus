"""JWT authentication middleware and token issuance (opt-in).

Auth is disabled unless `AUTH_ENABLED=true` AND `JWT_SECRET` are set.
Client credentials for the token endpoint come from AUTH_CLIENT_ID /
AUTH_CLIENT_SECRET; without them the endpoint refuses to issue tokens.
"""

from datetime import UTC, datetime, timedelta

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

EXEMPT_PATHS = frozenset({
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/v1/health",
    "/v1/meta",
    "/favicon.ico",
    "/v1/auth/token",
    "/v1/metrics",
})


def auth_active() -> bool:
    return settings.auth_enabled and bool(settings.jwt_secret)


def get_subject(request: Request) -> str | None:
    """Return the authenticated subject, or None when anonymous.

    Reads what JWTAuthMiddleware stored on request.state. No fallback
    here so callers choose explicitly between hard 401 and the
    single-user "local" placeholder.
    """
    subject = getattr(request.state, "subject", None)
    if subject:
        return str(subject)
    return None


def resolve_subject(request: Request) -> str:
    """Server-side identity with the P7 single-user fallback.

    Authenticated subject wins. Anonymous callers get "local" only when
    AUTH_SINGLE_USER_MODE is on. Otherwise empty string, which P7-1
    treats as unauthenticated. "local" stays a placeholder, never proof
    of authorization.
    """
    subject = get_subject(request)
    if subject:
        return subject
    if settings.auth_single_user_mode:
        return "local"
    return ""


def create_access_token(subject: str) -> tuple[str, int]:
    expires_in = settings.access_token_expire_minutes * 60
    now = datetime.now(UTC)
    payload: dict[str, object] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
    }
    if settings.jwt_issuer:
        payload["iss"] = settings.jwt_issuer
    assert settings.jwt_secret is not None
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


async def require_subject(request: Request) -> str:
    """FastAPI dependency for P7-1. Safe to import now, enforced later.

    When auth is off, returns the single-user fallback without touching
    headers. When auth is on, missing or empty subject means 401. Respects
    AUTH_REQUIRE_AUTH only as a future cutover flag; default off keeps
    current open behavior.
    """
    from fastapi import HTTPException

    if not auth_active():
        return resolve_subject(request)
    subject = get_subject(request)
    if subject:
        return subject
    if not settings.auth_require_auth and settings.auth_single_user_mode:
        return "local"
    raise HTTPException(status_code=401, detail="Not authenticated.")


class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not auth_active() or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            decode_kwargs: dict[str, object] = {
                "algorithms": [settings.jwt_algorithm],
            }
            if settings.jwt_issuer:
                decode_kwargs["issuer"] = settings.jwt_issuer
            payload = jwt.decode(
                token,
                settings.jwt_secret or "",
                **decode_kwargs,  # type: ignore[arg-type]
            )
        except jwt.PyJWTError as exc:
            logger.info({"message": "Rejected invalid token", "error": str(exc)})
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.subject = payload.get("sub", "")
        return await call_next(request)
