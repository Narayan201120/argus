"""P5-1 live-runner guard + row-validator tests (DEC-054, no network).

Never touches the network: httpx.Client is replaced with a stub that raises
on instantiation, so any attempted I/O fails the test loudly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts import eval_live
from scripts.eval_live import EvalInfraError, estimate_per_run_usd, validate_live_row

CORPUS = "evals/corpus.yaml"


def _good_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": "radar-01",
        "investigation_id": "inv_abc",
        "status": "complete",
        "status_reason": None,
        "counts": {"evidence": 2, "claims": 1},
        "evidence_ids": ["ev-1", "ev-2"],
        "claims": [
            {
                "id": "cl-1",
                "statement": "canned answer",
                "confidence": 0.8,
                "evidence_ids": ["ev-1", "ev-2"],
                "status": "supported",
            }
        ],
        "syntheses": [{"milestone": 0, "final": True}],
        "elapsed_s": 12.5,
        "timed_out": False,
    }
    row.update(overrides)
    return row


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("network attempted in a no-network test")

    monkeypatch.setattr(httpx, "Client", _ExplodingClient)


def _locked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_EVAL_LIVE", raising=False)


# ── Guard ───────────────────────────────────────────────────────────────────


def test_no_unlock_refuses_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    _no_network(monkeypatch)
    assert eval_live.main(["--corpus", CORPUS, "--only", "radar-01"]) == 2
    assert "Refusing to run" in capsys.readouterr().err


def test_env_unlock_still_blocked_by_spend_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    _no_network(monkeypatch)
    monkeypatch.setenv("ARGUS_EVAL_LIVE", "1")
    assert eval_live.main(["--corpus", CORPUS, "--spend-cap-usd", "0.01"]) == 2
    assert "cap" in capsys.readouterr().err


def test_dry_run_needs_no_unlock_and_no_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    _no_network(monkeypatch)
    assert eval_live.main(["--corpus", CORPUS, "--only", "radar-01", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 investigate POST(s) planned" in out
    assert "ceiling estimate" in out


def test_dry_run_over_cap_still_exits_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    _no_network(monkeypatch)
    assert eval_live.main(["--corpus", CORPUS, "--dry-run", "--spend-cap-usd", "0.01"]) == 0
    assert "would abort" in capsys.readouterr().out


def test_spend_estimate_honest_about_inputs() -> None:
    from app.config import settings

    expected = settings.max_web_calls * settings.cost_usd_per_web_search + eval_live.EST_LLM_USD_PER_RUN
    assert estimate_per_run_usd() == pytest.approx(expected)
    assert estimate_per_run_usd() > 0


def test_unknown_case_id_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert eval_live.main(["--corpus", CORPUS, "--dry-run", "--only", "nope-99"]) == 3
    assert "Unknown case ids" in capsys.readouterr().err


def test_run_one_posts_202_and_polls_to_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.eval.corpus import load_corpus

    case = next(c for c in load_corpus(CORPUS) if c.id == "radar-01")
    calls: list[str] = []

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    polls = iter([{"status": "gathering"}, {"status": "gathering"}, _terminal_board()])

    class _FakeClient:
        def post(self, path: str, **kwargs: Any) -> _FakeResponse:
            calls.append(f"POST {path}")
            assert kwargs["json"] == {"query": case.query}
            return _FakeResponse(202, {"investigation_id": "inv_1"})

        def get(self, path: str, **kwargs: Any) -> _FakeResponse:
            calls.append(f"GET {path}")
            assert path == "/v1/investigate/inv_1"
            return _FakeResponse(200, next(polls))

    monkeypatch.setattr(eval_live.time, "sleep", lambda _s: None)
    row = eval_live.run_one(_FakeClient(), case, poll_s=0.01, timeout_s=60.0)  # type: ignore[arg-type]
    assert row["case_id"] == "radar-01"
    assert row["investigation_id"] == "inv_1"
    assert row["status"] == "complete"
    assert row["timed_out"] is False
    assert row["counts"] == {"evidence": 2, "claims": 1}
    assert calls[0] == "POST /v1/investigate"
    assert len([c for c in calls if c.startswith("GET")]) == 3


def test_run_one_non_202_is_infra_error() -> None:
    class _FakeResponse:
        status_code = 500
        text = "boom"

    class _FakeClient:
        def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse()

    from app.eval.corpus import load_corpus

    case = load_corpus(CORPUS)[0]
    with pytest.raises(EvalInfraError, match="HTTP 500"):
        eval_live.run_one(_FakeClient(), case, poll_s=0.01, timeout_s=1.0)  # type: ignore[arg-type]


def _terminal_board() -> dict[str, Any]:
    return {
        "status": "complete",
        "evidence": [{"id": "ev-1"}, {"id": "ev-2"}],
        "claims": [
            {
                "id": "cl-1",
                "statement": "canned",
                "confidence": 0.8,
                "evidence_ids": ["ev-1"],
                "status": "supported",
            }
        ],
        "syntheses": [{"milestone": 0, "final": True}],
    }


# ── validate_live_row (DEC-054: verdicts never constrain terminal state) ────


def test_complete_with_final_synthesis_passes() -> None:
    features, failures = validate_live_row(_good_row())
    assert failures == []
    assert features["terminal"] == "complete"
    assert features["claim_count"] == 1
    assert features["grounded_ratio"] == pytest.approx(1.0)
    assert features["has_final_synthesis"] is True


def test_non_terminal_status_fails() -> None:
    _, failures = validate_live_row(_good_row(status="gathering"))
    assert any("not terminal" in failure for failure in failures)


def test_complete_without_final_synthesis_fails() -> None:
    _, failures = validate_live_row(_good_row(syntheses=[{"milestone": 0, "final": False}]))
    assert any("final synthesis" in failure for failure in failures)


def test_dangling_citation_fails_grounding() -> None:
    row = _good_row()
    assert isinstance(row["claims"], list)
    row["claims"][0]["evidence_ids"] = ["ev-missing"]
    _, failures = validate_live_row(row)
    assert any("grounding" in failure for failure in failures)


def test_supported_without_evidence_fails_honest_gap() -> None:
    row = _good_row(evidence_ids=[])
    assert isinstance(row["claims"], list)
    row["claims"][0]["evidence_ids"] = []
    _, failures = validate_live_row(row)
    assert failures


def test_refused_correctly_passes_in_any_terminal_state() -> None:
    for terminal in ("complete", "failed", "budget_exhausted", "cancelled"):
        row = _good_row(
            status=terminal,
            evidence_ids=[],
            claims=[],
            counts={"evidence": 0, "claims": 0},
            syntheses=[{"milestone": 0, "final": terminal == "complete"}],
        )
        features, failures = validate_live_row(row)
        # Empty refusal board (final refusal synthesis on complete): clean
        # in every terminal state per DEC-054.
        assert failures == [], (terminal, failures)
        assert features["refused_correctly"] is True


def test_live_rows_serialize_as_jsonl(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    features, failures = validate_live_row(_good_row())
    row = _good_row(features=features, failures=failures)
    target.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = json.loads(target.read_text(encoding="utf-8").strip())
    assert loaded["case_id"] == "radar-01"
    assert loaded["failures"] == []
