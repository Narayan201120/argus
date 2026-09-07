import pytest
from fastapi.testclient import TestClient

from app.connectors import availability
from app.connectors.availability import (
    consecutive_auth_failures,
    is_auth_failure,
    is_demoted,
    record_auth_failure,
    record_success,
)
from app.connectors.base import ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app
from app.orchestration.binding import RoleBindingService, RoutingConfig
from app.orchestration.workers import run_connector_query

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_availability():
    availability.reset_all()
    yield
    availability.reset_all()


class _Stub:
    def __init__(self, connector_id: str):
        self.connector_id = connector_id
        self.display_name = f"{connector_id.title()} Stub"
        self.capabilities = ["text"]
        self.is_available = True


def _resp(status: ConnectorStatus, error: str | None = None) -> ConnectorResponse:
    return ConnectorResponse(
        model_id="m",
        content="" if status != ConnectorStatus.SUCCESS else "ok",
        latency_ms=1,
        token_usage=TokenUsage(),
        status=status,
        error=error,
    )


# --- classification ---


def test_401_counts_as_auth_failure():
    assert is_auth_failure("error", "401 Unauthorized") is True


def test_unauthorized_and_invalid_key_count():
    assert is_auth_failure("error", "unauthorized") is True
    assert is_auth_failure("error", "invalid_api_key: check your key") is True
    assert is_auth_failure("error", "Invalid API key provided") is True


def test_429_does_not_count():
    assert is_auth_failure("rate_limited", "429 rate limit exceeded") is False
    assert is_auth_failure("error", "429 You exceeded your current quota.") is False


def test_500_does_not_count():
    assert is_auth_failure("error", "500 Internal Server Error") is False


def test_timeout_does_not_count():
    assert is_auth_failure("timeout", "Timed out after 45s") is False
    assert is_auth_failure("error", "request timeout after 45s") is False


def test_quota_does_not_count():
    assert is_auth_failure("error", "You exceeded your current quota.") is False


def test_success_never_counts():
    assert is_auth_failure("success", None) is False


# --- 3-strikes + reset ---


def test_three_strikes_demotes():
    assert is_demoted("openai") is False
    record_auth_failure("openai")
    record_auth_failure("openai")
    assert is_demoted("openai") is False
    record_auth_failure("openai")
    assert is_demoted("openai") is True
    assert consecutive_auth_failures("openai") == 3


def test_success_resets_counter():
    record_auth_failure("openai")
    record_auth_failure("openai")
    record_success("openai")
    assert consecutive_auth_failures("openai") == 0
    assert is_demoted("openai") is False


def test_success_restores_demoted():
    for _ in range(3):
        record_auth_failure("claude")
    assert is_demoted("claude") is True
    record_success("claude")
    assert is_demoted("claude") is False


# --- run_connector_query wiring ---


class _AuthFailConnector(_Stub):
    async def query(self, prompt, sub_query, config):
        return _resp(ConnectorStatus.ERROR, "401 invalid_api_key")

    async def health_check(self):
        return True


class _OkConnector(_Stub):
    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id="m", content="ok", latency_ms=1,
            token_usage=TokenUsage(), status=ConnectorStatus.SUCCESS,
        )

    async def health_check(self):
        return True


class _QuotaConnector(_Stub):
    async def query(self, prompt, sub_query, config):
        return _resp(ConnectorStatus.RATE_LIMITED, "429 quota exceeded")

    async def health_check(self):
        return True


async def test_run_connector_query_records_auth_failure_and_success():
    bad = _AuthFailConnector("openai")
    for _ in range(3):
        await run_connector_query(bad, "p", "s", ConnectorConfig())
    assert is_demoted("openai") is True

    ok = _OkConnector("openai")
    await run_connector_query(ok, "p", "s", ConnectorConfig())
    assert is_demoted("openai") is False


async def test_run_connector_query_ignores_quota():
    q = _QuotaConnector("mistral")
    for _ in range(5):
        await run_connector_query(q, "p", "s", ConnectorConfig())
    assert consecutive_auth_failures("mistral") == 0
    assert is_demoted("mistral") is False


# --- selection ---


def _service() -> RoleBindingService:
    return RoleBindingService(RoutingConfig())


def test_selection_skips_demoted():
    svc = _service()
    active = [_Stub("openai"), _Stub("mistral")]
    for _ in range(3):
        record_auth_failure("openai")
    picked = svc.select_connector(active, "analyzer")
    assert picked.connector_id == "mistral"


def test_all_demoted_falls_through_without_raising():
    svc = _service()
    active = [_Stub("openai"), _Stub("mistral")]
    for cid in ("openai", "mistral"):
        for _ in range(3):
            record_auth_failure(cid)
    picked = svc.select_connector(active, "analyzer")
    assert picked.connector_id in {"openai", "mistral"}


# --- /v1/models visibility ---


def test_models_output_carries_demoted_fields(monkeypatch):
    monkeypatch.setattr(registry, "_connectors", {"openai": _Stub("openai")})
    data = client.get("/v1/models").json()
    entry = next(c for c in data["connectors"] if c["connector_id"] == "openai")
    assert entry["demoted"] is False
    assert entry["consecutive_auth_failures"] == 0

    for _ in range(3):
        record_auth_failure("openai")
    data = client.get("/v1/models").json()
    entry = next(c for c in data["connectors"] if c["connector_id"] == "openai")
    assert entry["demoted"] is True
    assert entry["consecutive_auth_failures"] == 3


# ── Analysis worker demotion wiring ─────────────────────────────────────────


def test_worker_pick_skips_demoted(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.analysis.workers as workers_module

    dead = _Stub("dead-llm")
    live = _Stub("live-llm")
    for _ in range(3):
        record_auth_failure("dead-llm")
    monkeypatch.setattr(workers_module.registry, "available", lambda: [dead, live])
    assert workers_module._pick_connector() is live


def test_worker_pick_pinned_demoted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.analysis.workers as workers_module
    from app.analysis.workers import WorkerError
    from app.config import settings

    dead = _Stub("dead-llm")
    for _ in range(3):
        record_auth_failure("dead-llm")
    monkeypatch.setattr(workers_module.registry, "get", lambda _id: dead)
    monkeypatch.setattr(settings, "analysis_connector_id", "dead-llm")
    with pytest.raises(WorkerError, match="demoted"):
        workers_module._pick_connector()


def test_record_outcome_tracks_auth_and_success() -> None:
    import app.analysis.workers as workers_module

    bad = _resp(ConnectorStatus.ERROR, "401 Unauthorized")
    workers_module._record_outcome("c9", bad)
    assert consecutive_auth_failures("c9") == 1
    ok = _resp(ConnectorStatus.SUCCESS)
    workers_module._record_outcome("c9", ok)
    assert consecutive_auth_failures("c9") == 0
    assert is_demoted("c9") is False
