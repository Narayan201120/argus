"""P5-1 diff gate + normalization tests (DEC-054, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.eval.diff import CaseEntry, EvalBaseline, diff_runs, p90
from app.eval.normalize import NORMALIZE_TRUNCATE_CHARS, board_summary, normalize_report
from scripts import eval_diff


def _entry(grounded: float = 1.0, elapsed: float | None = 10.0,
           failures: list[str] | None = None, final: bool = True) -> CaseEntry:
    return CaseEntry(
        terminal="complete",
        claim_count=0 if grounded == 1.0 else 1,
        grounded_ratio=grounded,
        has_final_synthesis=final,
        refused_correctly=grounded == 1.0,
        elapsed_s=elapsed,
        failures=list(failures or []),
    )


def _run(cases: dict[str, CaseEntry], version: str = "test") -> EvalBaseline:
    return EvalBaseline(version=version, cases=cases)


# ── diff green/red ──────────────────────────────────────────────────────────


def test_identical_runs_pass_with_zero_deltas() -> None:
    cases = {f"case-{i}": _entry() for i in range(5)}
    diff = diff_runs(_run(cases), _run(cases))
    assert diff.passed is True
    assert diff.verdict_drops == 0
    assert diff.grounding_delta_pp == 0.0
    assert diff.p90_final_report_delta_pct == 0.0
    assert diff.compared_cases == 5


def test_verdict_drop_fails() -> None:
    base = _run({"a": _entry(), "b": _entry()})
    cand = _run({"a": _entry(), "b": _entry(failures=["grounding: dangling"])})
    diff = diff_runs(base, cand)
    assert diff.verdict_drops == 1
    assert diff.passed is False


def test_fixed_case_is_not_a_drop() -> None:
    base = _run({"a": _entry(failures=["grounding: dangling"])})
    cand = _run({"a": _entry()})
    diff = diff_runs(base, cand)
    assert diff.verdict_drops == 0
    assert diff.passed is True


def test_grounding_drop_inside_threshold_passes() -> None:
    base = _run({f"c{i}": _entry(grounded=1.0, final=False) for i in range(40)})
    cand = _run({"c0": _entry(grounded=0.0, final=False),
                 **{f"c{i}": _entry(grounded=1.0, final=False) for i in range(1, 40)}})
    diff = diff_runs(base, cand)
    assert diff.grounding_delta_pp == pytest.approx(-2.5)
    assert diff.passed is True


def test_grounding_drop_beyond_threshold_fails() -> None:
    base = _run({f"c{i}": _entry(grounded=1.0, final=False) for i in range(10)})
    cand = _run({"c0": _entry(grounded=0.0, final=False),
                 **{f"c{i}": _entry(grounded=1.0, final=False) for i in range(1, 10)}})
    diff = diff_runs(base, cand)
    assert diff.grounding_delta_pp == pytest.approx(-10.0)
    assert diff.passed is False


def test_grounding_boundary_equal_passes_excess_fails() -> None:
    # Exact boundary logic without float noise: with a zero threshold, a zero
    # delta is at the boundary and passes; any real drop fails. (A -5.00pp
    # delta built from 0.95 ratios is -5.000000000000004 in binary floating
    # point, so the default-threshold edge is covered by convention here.)
    from app.eval.diff import DEFAULT_MAX_GROUNDING_DROP_PP, DEFAULT_MAX_P90_DELTA_PCT

    assert DEFAULT_MAX_GROUNDING_DROP_PP == 5.0
    assert DEFAULT_MAX_P90_DELTA_PCT == 25.0
    base = _run({"a": _entry(grounded=1.0, final=False)})
    same = _run({"a": _entry(grounded=1.0, final=False)})
    assert diff_runs(base, same, max_grounding_drop_pp=0.0).passed is True
    tiny_drop = _run({"a": _entry(grounded=0.999999, final=False)})
    assert diff_runs(base, tiny_drop, max_grounding_drop_pp=0.0).passed is False


def test_p90_regression_boundary() -> None:
    base = _run({f"c{i}": _entry(elapsed=100.0) for i in range(10)})
    cand_ok = _run({f"c{i}": _entry(elapsed=125.0) for i in range(10)})
    diff_ok = diff_runs(base, cand_ok)
    assert diff_ok.p90_final_report_delta_pct == 25.0
    assert diff_ok.passed is True
    cand_bad = _run({f"c{i}": _entry(elapsed=125.01) for i in range(10)})
    diff_bad = diff_runs(base, cand_bad)
    assert diff_bad.p90_final_report_delta_pct > 25.0
    assert diff_bad.passed is False


def test_empty_baseline_trivially_passes() -> None:
    empty = EvalBaseline.model_validate(json.loads(Path("evals/baselines/v0.6.0.json").read_text()))
    assert empty.version == "v0.6.0"
    assert empty.cases == {}
    cand = _run({"a": _entry(failures=["anything"])})
    diff = diff_runs(empty, cand)
    assert diff.compared_cases == 0
    assert diff.passed is True


def test_disjoint_cases_compare_nothing_and_pass() -> None:
    diff = diff_runs(_run({"a": _entry()}), _run({"b": _entry(failures=["x"])}))
    assert diff.compared_cases == 0
    assert diff.passed is True


def test_p90_without_timings_is_zero() -> None:
    base = _run({"a": _entry(elapsed=None, final=False)})
    cand = _run({"a": _entry(elapsed=None, final=False)})
    assert diff_runs(base, cand).p90_final_report_delta_pct == 0.0


def test_p90_nearest_rank() -> None:
    assert p90([]) is None
    assert p90([5.0]) == 5.0
    assert p90([float(v) for v in range(1, 11)]) == 9.0


# ── normalize ───────────────────────────────────────────────────────────────


def test_normalize_report_shapes_text() -> None:
    assert normalize_report("Hello,   World!") == "hello world"
    assert normalize_report("  A\n\nB\tC  ") == "a b c"
    long_text = "x" * (NORMALIZE_TRUNCATE_CHARS + 100)
    assert len(normalize_report(long_text)) == NORMALIZE_TRUNCATE_CHARS


def test_normalize_never_requires_exact_match() -> None:
    # Same content, different surface form: normalization converges them, so
    # the gate never depends on byte equality.
    assert normalize_report("# Report: Foo (v2).") == normalize_report("report  foo v2")


def test_board_summary_grounded_board() -> None:
    row: dict[str, Any] = {
        "status": "complete",
        "evidence_ids": ["ev-1"],
        "claims": [{"id": "cl-1", "evidence_ids": ["ev-1"], "status": "supported"}],
        "syntheses": [{"milestone": 0, "final": True}],
    }
    summary = board_summary(row)
    assert summary == {
        "terminal": "complete",
        "claim_count": 1,
        "grounded_ratio": 1.0,
        "has_final_synthesis": True,
        "refused_correctly": False,
    }


def test_board_summary_empty_board_is_refused_correctly() -> None:
    row: dict[str, Any] = {
        "status": "budget_exhausted",
        "evidence_ids": [],
        "claims": [],
        "syntheses": [],
    }
    summary = board_summary(row)
    assert summary["refused_correctly"] is True
    assert summary["grounded_ratio"] == 1.0
    assert summary["has_final_synthesis"] is False


def test_board_summary_proposed_empty_claims_count_as_refusal() -> None:
    row: dict[str, Any] = {
        "status": "failed",
        "evidence_ids": [],
        "claims": [{"id": "cl-1", "evidence_ids": [], "status": "proposed"}],
        "syntheses": [],
    }
    assert board_summary(row)["refused_correctly"] is True


def test_board_summary_supported_empty_claim_is_fabrication() -> None:
    row: dict[str, Any] = {
        "status": "complete",
        "evidence_ids": [],
        "claims": [{"id": "cl-1", "evidence_ids": [], "status": "supported"}],
        "syntheses": [{"milestone": 0, "final": True}],
    }
    summary = board_summary(row)
    assert summary["refused_correctly"] is False
    assert summary["grounded_ratio"] == 0.0


# ── CLI ─────────────────────────────────────────────────────────────────────


def _write(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _candidate_row(case_id: str, grounded: bool = True) -> dict[str, Any]:
    evidence = ["ev-1"] if grounded else []
    return {
        "case_id": case_id,
        "investigation_id": "inv_x",
        "status": "complete",
        "status_reason": None,
        "counts": {"evidence": 1, "claims": 1},
        "evidence_ids": evidence,
        "claims": [
            {"id": "cl-1", "statement": "s", "confidence": 0.8,
             "evidence_ids": evidence if grounded else ["ev-gone"], "status": "supported"}
        ],
        "syntheses": [{"milestone": 0, "final": True}],
        "elapsed_s": 10.0,
        "timed_out": False,
    }


def test_cli_green_and_red(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "base.json", {
        "version": "vX",
        "cases": {
            "a": {"terminal": "complete", "claim_count": 1, "grounded_ratio": 1.0,
                  "has_final_synthesis": True, "refused_correctly": False,
                  "elapsed_s": 10.0, "failures": []},
        },
    })
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(_candidate_row("a")) + "\n", encoding="utf-8")
    assert eval_diff.main(["--baseline", baseline, "--candidate", str(good)]) == 0

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(_candidate_row("a", grounded=False)) + "\n", encoding="utf-8")
    assert eval_diff.main(["--baseline", baseline, "--candidate", str(bad)]) == 1


def test_cli_usage_errors(tmp_path: Path) -> None:
    assert eval_diff.main([]) == 2
    assert eval_diff.main(["--candidate", str(tmp_path / "missing.jsonl")]) == 2
    baseline = _write(tmp_path / "base.json", {"version": "v", "cases": {}})
    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps(_candidate_row("a")) + "\n", encoding="utf-8")
    assert eval_diff.main(["--baseline", baseline, "--candidate", str(good),
                           "--fail-on", "nope"]) == 2


def test_cli_gate_selection_disables_verdict_drop(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "base.json", {
        "version": "v",
        "cases": {
            "a": {"terminal": "complete", "claim_count": 1, "grounded_ratio": 1.0,
                  "has_final_synthesis": True, "refused_correctly": False,
                  "elapsed_s": 10.0, "failures": []},
        },
    })
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(_candidate_row("a", grounded=False)) + "\n", encoding="utf-8")
    assert eval_diff.main(["--baseline", baseline, "--candidate", str(bad),
                           "--fail-on", "p90-regression"]) == 0
