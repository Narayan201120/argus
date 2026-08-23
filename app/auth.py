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
    "/v1/auth/token",
})


def auth_active() -> bool:
    return settings.auth_enabled and bool(settings.jwt_secret)


def create_access_token(subject: str) -> tuple[str, int]:
    expires_in = settings.access_token_expire_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
    }
    assert settings.jwt_secret is not None
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


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
            payload = jwt.decode(
                token,
                settings.jwt_secret or "",
                algorithms=[settings.jwt_algorithm],
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
