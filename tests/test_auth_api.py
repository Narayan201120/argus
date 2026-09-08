import time

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "jwt_secret", "test-secret")
    monkeypatch.setattr(settings, "auth_client_id", "dev-client")
    monkeypatch.setattr(settings, "auth_client_secret", "dev-secret-rotate-me")
    with TestClient(app) as test_client:
        yield test_client


def _issue(client: auth_client, client_id="dev-client", client_secret="dev-secret-rotate-me"):
    return client.post("/v1/auth/token", json={
        "client_id": client_id, "client_secret": client_secret,
    })


def test_health_and_docs_exempt_without_token(auth_client):
    assert auth_client.get("/v1/health").status_code == 200
    assert auth_client.get("/").status_code == 200


def test_protected_route_rejects_missing_token(auth_client):
    response = auth_client.get("/v1/models")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_token_endpoint_requires_configured_credentials(auth_client):
    assert _issue(auth_client, client_secret="wrong").status_code == 401
    assert _issue(auth_client, client_id="nobody").status_code == 401


def test_full_token_roundtrip(auth_client):
    issued = _issue(auth_client)
    assert issued.status_code == 200
    token = issued.json()["access_token"]
    assert issued.json()["token_type"] == "bearer"
    assert issued.json()["expires_in"] == settings.access_token_expire_minutes * 60

    assert auth_client.get("/v1/models").status_code == 401
    ok = auth_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {token}"}
    )
    assert ok.status_code == 200


def test_expired_token_rejected(auth_client):
    past = int(time.time()) - 3600
    expired = pyjwt.encode(
        {"sub": "dev-client", "exp": past},
        settings.jwt_secret or "",
        algorithm=settings.jwt_algorithm,
    )
    response = auth_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {expired}"}
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_garbage_token_rejected(auth_client):
    response = auth_client.get(
        "/v1/models", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert response.status_code == 401


def test_auth_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.delattr(settings, "jwt_secret", raising=False)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 200


def test_meta_and_favicon_exempt_without_token(auth_client):
    assert auth_client.get("/v1/meta").status_code == 200
    assert auth_client.get("/favicon.ico").status_code == 204


def test_middleware_order_locked():
    # user_middleware lists outermost first, which is also execution order.
    names = [m.cls.__name__ for m in app.user_middleware]
    assert names == [
        "PrometheusMiddleware",
        "JWTAuthMiddleware",
        "RateLimitMiddleware",
    ]


def test_subject_visible_to_ratelimit():
    from starlette.requests import Request

    from app.ratelimit import identity_for

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/models",
        "headers": [],
        "client": ("9.9.9.9", 1234),
    }
    request = Request(scope)
    request.state.subject = "alice"
    assert identity_for(request) == "sub:alice"


def test_auth_me_reports_subject(auth_client):
    me = auth_client.get("/v1/auth/me")
    assert me.status_code == 401
    issued = _issue(auth_client)
    token = issued.json()["access_token"]
    me = auth_client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert me.status_code == 200
    assert me.json()["sub"] == "dev-client"


def test_auth_me_single_user_fallback_without_auth(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "auth_single_user_mode", True)
    with TestClient(app) as client:
        response = client.get("/v1/auth/me")
        assert response.status_code == 200
        assert response.json()["sub"] == "local"


def test_issuer_roundtrip_when_configured(auth_client, monkeypatch):
    import jwt as pyjwt

    monkeypatch.setattr(settings, "jwt_issuer", "argus-dev")
    issued = _issue(auth_client)
    token = issued.json()["access_token"]
    payload = pyjwt.decode(
        token,
        settings.jwt_secret or "",
        algorithms=[settings.jwt_algorithm],
        issuer="argus-dev",
    )
    assert payload["iss"] == "argus-dev"
    assert (
        auth_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 200
    )
