"""RAG exchange contract test (P7-2 Option C). Mock-only, zero network."""

import jwt as pyjwt
import pytest

from app.config import settings
from app.tools.rag import RAG_EXCHANGE_DEFAULT_PATH, exchange_url, mint_exchange_assertion

ARGUS_MINT_LIFETIME_S = 60
RAG_ACCEPT_MAX_LIFETIME_S = 120


def _mint_payload_no_verify(token: str) -> dict:
    return pyjwt.decode(token, options={"verify_signature": False})


def test_mint_claims_shape_and_lifetime(monkeypatch):
    monkeypatch.setattr(settings, "rag_exchange_secret", "test-exchange-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "argus-test")
    token = mint_exchange_assertion("alice")
    payload = _mint_payload_no_verify(token)
    assert payload["sub"] == "alice"
    assert payload["iss"] == "argus-test"
    assert payload["exp"] - payload["iat"] == ARGUS_MINT_LIFETIME_S == 60


def test_mint_verifies_with_shared_secret(monkeypatch):
    monkeypatch.setattr(settings, "rag_exchange_secret", "test-exchange-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "argus-test")
    token = mint_exchange_assertion("alice")
    payload = pyjwt.decode(token, "test-exchange-secret", algorithms=["HS256"])
    assert payload["sub"] == "alice"


def test_wrong_secret_verification_fails(monkeypatch):
    monkeypatch.setattr(settings, "rag_exchange_secret", "test-exchange-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "argus-test")
    token = mint_exchange_assertion("alice")
    with pytest.raises(pyjwt.PyJWTError):
        pyjwt.decode(token, "wrong-secret", algorithms=["HS256"])


def test_tampered_claim_fails_verification(monkeypatch):
    monkeypatch.setattr(settings, "rag_exchange_secret", "test-exchange-secret")
    monkeypatch.setattr(settings, "jwt_issuer", "argus-test")
    token = mint_exchange_assertion("alice")
    # Flip a char in the PAYLOAD segment: the last char of the whole token
    # sits in base64 padding bits and decodes to identical bytes, so it
    # would verify. Payload tampering must break the signature.
    header, payload, signature = token.split(".")
    tampered_payload = ("A" if payload[0] != "A" else "B") + payload[1:]
    tampered = f"{header}.{tampered_payload}.{signature}"
    with pytest.raises(pyjwt.PyJWTError):
        pyjwt.decode(tampered, "test-exchange-secret", algorithms=["HS256"])


def test_exchange_url_default_derivation(monkeypatch):
    assert RAG_EXCHANGE_DEFAULT_PATH == "/api/service/exchange/"
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.example:8001")
    monkeypatch.setattr(settings, "rag_exchange_url", None)
    assert exchange_url() == "http://rag.example:8001" + RAG_EXCHANGE_DEFAULT_PATH


def test_exchange_url_explicit_override(monkeypatch):
    monkeypatch.setattr(settings, "rag_exchange_url", "http://rag.example:8001/custom/exchange/")
    assert exchange_url() == "http://rag.example:8001/custom/exchange/"


def test_expiry_math_boundary():
    assert ARGUS_MINT_LIFETIME_S == 60
    assert RAG_ACCEPT_MAX_LIFETIME_S == 120
    assert ARGUS_MINT_LIFETIME_S <= RAG_ACCEPT_MAX_LIFETIME_S
