"""Stage 8 - live smoke script validators and execution guard."""

import pytest

from scripts import smoke_live


@pytest.fixture(autouse=True)
def _clear_live_unlock(monkeypatch):
    monkeypatch.delenv("ARGUS_SMOKE_LIVE", raising=False)


def test_main_refuses_to_run_without_unlock():
    assert smoke_live.main([]) == 2


def test_parser_defaults_and_only_filter():
    parser = smoke_live.build_parser()
    args = parser.parse_args([])
    assert args.base_url == smoke_live.DEFAULT_BASE_URL
    assert set(args.only) == set(smoke_live.ALL_CHECKS)

    scoped = parser.parse_args(["--only", "health", "models"])
    assert scoped.only == ["health", "models"]


def test_validate_health_accepts_ok_and_flags_bad_payloads():
    good = {"status": "ok", "connectors": [{"connector_id": "gemini"}], "redis": "disabled"}
    assert smoke_live.validate_health(good) == []

    degraded = {"status": "degraded", "connectors": [{"connector_id": "mistral"}]}
    assert smoke_live.validate_health(degraded) == []

    bad = {"status": "exploded"}
    failures = smoke_live.validate_health(bad)
    assert any("status" in f for f in failures)
    assert any("connectors" in f for f in failures)


def test_validate_models_requires_positive_total():
    assert smoke_live.validate_models({"connectors": [], "total": 2}) == []
    failures = smoke_live.validate_models({"total": 0})
    assert any("connectors" in f for f in failures)
    assert any("total" in f for f in failures)


def test_validate_query_response_checks_result_and_assignments():
    parallel = {
        "result": "answer",
        "short_circuited": False,
        "model_statuses": [{"role": "researcher"}],
        "role_assignments": {"researcher": "mistral"},
    }
    assert smoke_live.validate_query_response(parallel) == []

    direct = {
        "result": "answer",
        "short_circuited": True,
        "model_statuses": [],
        "role_assignments": {"direct": "claude"},
    }
    assert smoke_live.validate_query_response(direct) == []

    empty: dict = {}
    failures = smoke_live.validate_query_response(empty)
    assert any("result" in f for f in failures)
    assert any("role_assignments" in f for f in failures)


def test_validate_final_event_requires_router_strategy():
    envelope = {
        "result": "answer",
        "short_circuited": True,
        "model_statuses": [],
        "role_assignments": {"direct": "claude"},
        "router_strategy": "semantic",
    }
    assert smoke_live.validate_final_event(envelope) == []

    envelope["router_strategy"] = None
    failures = smoke_live.validate_final_event(envelope)
    assert any("router_strategy" in f for f in failures)
