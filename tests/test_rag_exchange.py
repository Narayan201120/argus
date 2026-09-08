"""P7-2 RAG per-user exchange client (Option C, DEC-056). Mock-only, zero live network."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import jwt
import pytest
from fakeredis import aioredis as fakeredis_aioredis

from app.config import settings
from app.tools.base import BaseTool, ToolResult
from app.tools.rag import (
    _ASSERTION_TTL_S,
    RAG_EXCHANGE_DEFAULT_PATH,
    RagRetrieveTool,
    exchange_url,
    mint_exchange_assertion,
)


@pytest.fixture
async def fake_redis() -> AsyncIterator[Any]:
    fr = fakeredis_aioredis.FakeRedis(decode_responses=True)
    yield fr
    await fr.aclose()


@pytest.fixture(autouse=True)
def _use_fake_redis(fake_redis: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rediskit import holder

    monkeypatch.setattr(holder, "client", fake_redis)


def _exchange_on(
    monkeypatch: pytest.MonkeyPatch,
    *,
    secret: str | None = "ex-secret",
    base_url: str | None = "http://rag.test",
    exchange_url_override: str | None = None,
    single_user: bool = False,
    issuer: str | None = None,
) -> None:
    monkeypatch.setattr(settings, "rag_integration_enabled", True)
    monkeypatch.setattr(settings, "rag_base_url", base_url)
    monkeypatch.setattr(settings, "rag_service_user", None)
    monkeypatch.setattr(settings, "rag_service_pass", None)
    monkeypatch.setattr(settings, "rag_exchange_enabled", True)
    monkeypatch.setattr(settings, "rag_exchange_secret", secret)
    monkeypatch.setattr(settings, "rag_exchange_url", exchange_url_override)
    monkeypatch.setattr(settings, "auth_single_user_mode", single_user)
    monkeypatch.setattr(settings, "jwt_issuer", issuer)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)


def _legacy_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_integration_enabled", True)
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.test")
    monkeypatch.setattr(settings, "rag_service_user", "svc")
    monkeypatch.setattr(settings, "rag_service_pass", "pw")
    monkeypatch.setattr(settings, "rag_exchange_enabled", False)
    monkeypatch.setattr(settings, "rag_exchange_secret", None)
    monkeypatch.setattr(settings, "rag_exchange_url", None)
    monkeypatch.setattr(settings, "auth_single_user_mode", True)
    monkeypatch.setattr(settings, "jwt_issuer", None)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)


class _Resp:
    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        text: str = "ok",
        bad_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self._bad_json = bad_json

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("bad json")
        return self._payload


def test_constants() -> None:
    assert RAG_EXCHANGE_DEFAULT_PATH == "/api/service/exchange/"
    assert _ASSERTION_TTL_S == 60


def test_mint_assertion_claims_default_iss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_exchange_secret", "s3cr3t")
    monkeypatch.setattr(settings, "jwt_issuer", None)
    token = mint_exchange_assertion("alice")
    assert jwt.get_unverified_header(token)["alg"] == "HS256"
    payload = jwt.decode(
        token, "s3cr3t", algorithms=["HS256"], options={"verify_exp": False}
    )
    assert payload["sub"] == "alice"
    assert payload["iss"] == "argus"
    assert payload["exp"] - payload["iat"] == 60


def test_mint_assertion_iss_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_exchange_secret", "s3cr3t")
    monkeypatch.setattr(settings, "jwt_issuer", "my-issuer")
    token = mint_exchange_assertion("bob")
    payload = jwt.decode(
        token, "s3cr3t", algorithms=["HS256"], options={"verify_exp": False}
    )
    assert payload["iss"] == "my-issuer"
    assert payload["sub"] == "bob"
    assert payload["exp"] - payload["iat"] == 60


def test_mint_assertion_missing_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_exchange_secret", None)
    with pytest.raises(RuntimeError):
        mint_exchange_assertion("alice")


def test_exchange_url_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_exchange_url", None)
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.test")
    assert exchange_url() == "http://rag.test/api/service/exchange/"
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.test/")
    assert exchange_url() == "http://rag.test/api/service/exchange/"
    monkeypatch.setattr(settings, "rag_exchange_url", "http://custom/ex")
    assert exchange_url() == "http://custom/ex"
    monkeypatch.setattr(settings, "rag_exchange_url", None)
    monkeypatch.setattr(settings, "rag_base_url", None)
    assert exchange_url() == "/api/service/exchange/"


@pytest.mark.asyncio
async def test_exchange_caches_per_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    _exchange_on(monkeypatch)
    calls: dict[str, Any] = {"n": 0, "assertions": []}

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> _Resp:
            assert url == "http://rag.test/api/service/exchange/"
            assertion = (kwargs.get("json") or {}).get("assertion")
            assert isinstance(assertion, str)
            calls["assertions"].append(assertion)
            calls["n"] += 1
            return _Resp({"tokens": {"access": f"tok-{calls['n']}"}, "expires_in": 1500})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    tool = RagRetrieveTool()
    first_alice = await tool._ensure_token("alice")
    first_bob = await tool._ensure_token("bob")
    assert calls["n"] == 2
    assert first_alice != first_bob
    # Subject of each assertion matches the caller.
    subs = [
        jwt.decode(a, "ex-secret", algorithms=["HS256"], options={"verify_exp": False})["sub"]
        for a in calls["assertions"]
    ]
    assert subs == ["alice", "bob"]
    # Repeats hit the per-subject cache: zero new POSTs.
    assert await tool._ensure_token("alice") == first_alice
    assert await tool._ensure_token("bob") == first_bob
    assert calls["n"] == 2
    assert set(tool._user_tokens) == {"alice", "bob"}


@pytest.mark.asyncio
async def test_exchange_parsing_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    shapes: list[tuple[Any, str]] = [
        ({"tokens": {"access": "t1"}}, "t1"),
        ({"access": "t2"}, "t2"),
        ({"access_token": "t3"}, "t3"),
        ({"token": "t4"}, "t4"),
        ({"tokens": {"access": "t5"}, "expires_in": 0}, "t5"),
        ({"tokens": {"access": "t6"}, "expires_in": -5}, "t6"),
        ({"tokens": {"access": "t7"}, "expires_in": "bad"}, "t7"),
    ]
    for payload, want in shapes:
        _exchange_on(monkeypatch)

        def _make_client(body: Any) -> Any:
            snapshot = dict(body)

            class _FakeClient:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    del args, kwargs

                async def __aenter__(self) -> "_FakeClient":
                    return self

                async def __aexit__(self, *args: Any) -> bool:
                    return False

                async def post(self, url: str, **kwargs: Any) -> _Resp:
                    del url, kwargs
                    return _Resp(dict(snapshot))

            return _FakeClient

        monkeypatch.setattr(httpx, "AsyncClient", _make_client(payload))
        tool = RagRetrieveTool()
        got = await tool._ensure_token("u")
        assert got == want
        # Missing/non-positive expires_in falls back to ~1500s.
        ttl = tool._user_tokens["u"][1] - time.time()
        assert 1400 < ttl <= 1501


@pytest.mark.asyncio
async def test_concurrent_same_subject_single_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    _exchange_on(monkeypatch)
    calls = {"n": 0}

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> _Resp:
            del url, kwargs
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return _Resp({"tokens": {"access": "shared"}, "expires_in": 1500})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    tool = RagRetrieveTool()
    results = await asyncio.gather(*[tool._ensure_token("alice") for _ in range(10)])
    assert results == ["shared"] * 10
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_legacy_single_user_sign_in_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _legacy_on(monkeypatch)
    calls: dict[str, Any] = {"n": 0, "bodies": [], "urls": []}

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> _Resp:
            calls["n"] += 1
            calls["urls"].append(url)
            calls["bodies"].append(kwargs.get("json"))
            return _Resp({"tokens": {"access": "legacy-tok"}, "expires_in": 1500})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    tool = RagRetrieveTool()
    # Default keeps old call sites working.
    assert await tool._ensure_token() == "legacy-tok"
    assert await tool._ensure_token("local") == "legacy-tok"
    assert calls["n"] == 1
    assert calls["urls"] == ["http://rag.test/api/sign-in/"]
    assert calls["bodies"] == [{"username": "svc", "password": "pw"}]
    assert tool._user_tokens == {}
    assert tool._access_token == "legacy-tok"


@pytest.mark.asyncio
async def test_multiuser_without_exchange_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "rag_integration_enabled", True)
    monkeypatch.setattr(settings, "rag_base_url", "http://rag.test")
    monkeypatch.setattr(settings, "rag_service_user", "svc")
    monkeypatch.setattr(settings, "rag_service_pass", "pw")
    monkeypatch.setattr(settings, "rag_exchange_enabled", False)
    monkeypatch.setattr(settings, "rag_exchange_secret", None)
    monkeypatch.setattr(settings, "rag_exchange_url", None)
    monkeypatch.setattr(settings, "auth_single_user_mode", False)
    monkeypatch.setattr(settings, "tool_timeout_s", 5)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no HTTP should fire without exchange")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    tool = RagRetrieveTool()
    with pytest.raises(RuntimeError, match="not configured"):
        await tool._ensure_token("alice")
    result = await tool.run("hello", {"owner": "alice"})
    assert result.ok is False
    assert result.error is not None and "not configured" in result.error


@pytest.mark.asyncio
async def test_401_retry_reexchanges_only_that_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exchange_on(monkeypatch)
    state: dict[str, Any] = {"exchanges": 0, "searches": 0}

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> _Resp:
            if url.endswith("/api/service/exchange/"):
                state["exchanges"] += 1
                assertion = (kwargs.get("json") or {}).get("assertion")
                assert isinstance(assertion, str)
                sub = jwt.decode(
                    assertion, "ex-secret", algorithms=["HS256"], options={"verify_exp": False}
                )["sub"]
                if state["exchanges"] <= 2:
                    return _Resp({"tokens": {"access": f"tok-{sub}-old"}, "expires_in": 1500})
                return _Resp({"tokens": {"access": f"tok-{sub}-new"}, "expires_in": 1500})
            assert url.endswith("/api/search/")
            state["searches"] += 1
            auth = (kwargs.get("headers") or {}).get("Authorization", "")
            if auth == "Bearer tok-alice-old":
                return _Resp(None, status_code=401, text="unauthorized")
            if auth == "Bearer tok-alice-new":
                return _Resp({"results": [{"chunk": "hello", "source": "s1"}]})
            raise AssertionError(f"unexpected search auth: {auth}")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    tool = RagRetrieveTool()
    assert await tool._ensure_token("alice") == "tok-alice-old"
    assert await tool._ensure_token("bob") == "tok-bob-old"
    assert state["exchanges"] == 2
    result = await tool.run("q", {"owner": "alice"})
    assert result.ok is True
    assert [i.source_ref for i in result.items] == ["s1"]
    assert state["exchanges"] == 3  # only alice re-exchanged
    assert state["searches"] == 2
    assert tool._user_tokens["alice"][0] == "tok-alice-new"
    assert tool._user_tokens["bob"][0] == "tok-bob-old"


@pytest.mark.asyncio
async def test_run_owner_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _exchange_on(monkeypatch)
    seen: list[str] = []

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> _Resp:
            if url.endswith("/api/service/exchange/"):
                assertion = (kwargs.get("json") or {}).get("assertion")
                assert isinstance(assertion, str)
                sub = jwt.decode(
                    assertion, "ex-secret", algorithms=["HS256"], options={"verify_exp": False}
                )["sub"]
                seen.append(sub)
                return _Resp({"tokens": {"access": f"tok-{sub}"}, "expires_in": 1500})
            return _Resp({"results": []})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    tool = RagRetrieveTool()
    await tool.run("q", {"owner": "alice"})
    await tool.run("q")
    await tool.run("q", {})
    await tool.run("q", {"owner": ""})
    # Repeats hit the cache: one exchange for alice, one for local.
    assert seen == ["alice", "local"]
    # Cached subjects do not re-exchange: alice + local only.
    assert sorted(tool._user_tokens) == ["alice", "local"]


def test_invalidate_subject() -> None:
    tool = RagRetrieveTool()
    tool._user_tokens = {"alice": ("a", time.time() + 100), "bob": ("b", time.time() + 100)}
    tool._access_token = "shared"
    tool._token_expires_at = time.time() + 100
    tool._invalidate_subject("alice")
    assert set(tool._user_tokens) == {"bob"}
    assert tool._access_token == "shared"  # other subjects keep the legacy token
    tool._invalidate_subject("local")
    assert tool._access_token is None
    assert tool._token_expires_at == 0.0


def test_enabled_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    def _enabled(
        *,
        integration: bool,
        base_url: str | None,
        user: str | None,
        pw: str | None,
        ex_on: bool,
        ex_secret: str | None,
    ) -> bool:
        monkeypatch.setattr(settings, "rag_integration_enabled", integration)
        monkeypatch.setattr(settings, "rag_base_url", base_url)
        monkeypatch.setattr(settings, "rag_service_user", user)
        monkeypatch.setattr(settings, "rag_service_pass", pw)
        monkeypatch.setattr(settings, "rag_exchange_enabled", ex_on)
        monkeypatch.setattr(settings, "rag_exchange_secret", ex_secret)
        return RagRetrieveTool().enabled

    assert _enabled(integration=False, base_url="http://r", user="u", pw="p", ex_on=False, ex_secret=None) is False
    assert _enabled(integration=True, base_url=None, user="u", pw="p", ex_on=False, ex_secret=None) is False
    assert _enabled(integration=True, base_url="http://r", user="u", pw="p", ex_on=False, ex_secret=None) is True
    assert _enabled(integration=True, base_url="http://r", user=None, pw=None, ex_on=True, ex_secret="s") is True
    assert _enabled(integration=True, base_url="http://r", user=None, pw=None, ex_on=True, ex_secret=None) is False
    assert _enabled(integration=True, base_url="http://r", user=None, pw=None, ex_on=False, ex_secret="s") is False
    assert _enabled(integration=True, base_url="http://r", user=None, pw=None, ex_on=False, ex_secret=None) is False
    assert _enabled(integration=True, base_url="http://r", user="u", pw="p", ex_on=True, ex_secret="s") is True


@pytest.mark.asyncio
async def test_run_one_passes_params() -> None:
    from app.tools.dispatch import _run_one

    class RecTool(BaseTool):
        name: str = "rec"

        def __init__(self) -> None:
            self.seen: Any = "unset"

        @property
        def enabled(self) -> bool:
            return True

        async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
            del query
            self.seen = params
            return ToolResult(tool_name=self.name, ok=True)

    tool = RecTool()
    await _run_one(tool, "q", {"owner": "alice"})
    assert tool.seen == {"owner": "alice"}
    await _run_one(tool, "q")
    assert tool.seen is None


@pytest.mark.asyncio
async def test_race_tools_and_planned_pass_params(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.tools import dispatch as dispatch_module

    class RecTool(BaseTool):
        name: str = "rec"

        def __init__(self) -> None:
            self.seen: Any = "unset"

        @property
        def enabled(self) -> bool:
            return True

        async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
            del query
            self.seen = params
            return ToolResult(tool_name=self.name, ok=True)

    tool = RecTool()
    deadline = time.time() + 5
    await dispatch_module._race_tools("race-probe", [tool], "q", deadline, {"owner": "bob"})
    assert tool.seen == {"owner": "bob"}
    tool2 = RecTool()
    await dispatch_module._race_planned(
        "race-probe", [(tool2, "q")], deadline, {"owner": "carol"}
    )
    assert tool2.seen == {"owner": "carol"}
    # Defaults preserve old behavior.
    tool3 = RecTool()
    await dispatch_module._race_tools("race-probe", [tool3], "q", deadline)
    assert tool3.seen is None


@pytest.mark.asyncio
async def test_run_tool_round_passes_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.evidence.store import EvidenceBoardStore
    from app.investigations import InvestigationManager
    from app.tools import dispatch as dispatch_module

    mgr = InvestigationManager(EvidenceBoardStore())
    monkeypatch.setattr(dispatch_module, "manager", mgr)

    seen: dict[str, Any] = {}

    class RecTool(BaseTool):
        name: str = "recorder"

        @property
        def enabled(self) -> bool:
            return True

        async def run(self, query: str, params: dict[str, Any] | None = None) -> ToolResult:
            del query
            seen["params"] = params
            return ToolResult(tool_name=self.name, ok=True)

    monkeypatch.setattr(
        dispatch_module, "build_tool_registry", lambda: {"recorder": RecTool()}
    )
    inv = await mgr.create("owner probe query", "alice-123")
    try:
        written, attempted, stopped, succeeded = await dispatch_module.run_tool_round(
            inv.id, [("recorder", "q")]
        )
        assert seen["params"] == {"owner": "alice-123"}
        assert (attempted, stopped, succeeded) == (True, False, True)
        assert written == 0
    finally:
        await mgr.cancel(inv.id)
