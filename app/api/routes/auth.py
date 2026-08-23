from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.auth import create_access_token
from app.config import settings

router = APIRouter()


class TokenRequest(BaseModel):
    client_id: str
    client_secret: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/auth/token", response_model=TokenResponse)
async def issue_token(request: TokenRequest) -> TokenResponse:
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="Token issuance is not configured.")
    if (
        not settings.auth_client_id
        or not settings.auth_client_secret
        or request.client_id != settings.auth_client_id
        or request.client_secret != settings.auth_client_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid client credentials.")

    token, expires_in = create_access_token(subject=request.client_id)
    return TokenResponse(access_token=token, expires_in=expires_in)
