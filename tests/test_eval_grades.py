"""P5-2 human grading tests (DEC-054, no network, no LLM judge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.eval.grades import (
    GRADE_VERDICTS,
    GradeJoin,
    GradeRecord,
    join_grades,
    parse_verdict_token,
)
from scripts import eval_grade


def _row(
    case_id: str,
    grounded: bool = True,
    investigation_id: str = "inv-1",
) -> dict[str, Any]:
    evidence = ["ev-1"] if grounded else []
    return {
        "case_id": case_id,
        "investigation_id": investigation_id,
        "status": "complete",
        "counts": {"evidence": 1, "claims": 1},
        "evidence_ids": evidence,
        "claims": [
            {
                "id": "cl-1",
                "statement": "canned answer",
                "confidence": 0.8,
                "evidence_ids": evidence if grounded else ["ev-gone"],
                "status": "supported",
            }
        ],
        "syntheses": [{"milestone": 0, "final": True}],
        "elapsed_s": 10.0,
        "timed_out": False,
    }


def _write_run(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


# ── GradeRecord ───────────────────────────────────────────────────────────


def test_grade_record_accepts_all_verdicts() -> None:
    for verdict in GRADE_VERDICTS:
        record = GradeRecord(
            case_id="a", investigation_id="inv-1", verdict=verdict,
            notes="", graded_at="2026-09-08T00:00:00+00:00",
        )
        assert record.verdict == verdict


def test_grade_record_rejects_bad_verdict() -> None:
    with pytest.raises(ValidationError):
        GradeRecord(case_id="a", investigation_id="inv-1",
                    verdict="mostly-fine", graded_at="2026-09-08T00:00:00+00:00")  # type: ignore[arg-type]


def test_grade_record_rejects_empty_case_id() -> None:
    with pytest.raises(ValidationError):
        GradeRecord(case_id="", investigation_id="inv-1",
                    verdict="supported", graded_at="2026-09-08T00:00:00+00:00")


def test_parse_verdict_token_shortcuts_and_names() -> None:
    assert parse_verdict_token("s") == "supported"
    assert parse_verdict_token("p") == "partial"
    assert parse_verdict_token("u") == "unsupported"
    assert parse_verdict_token("r") == "refused-correctly"
    assert parse_verdict_token("Supported") == "supported"
    assert parse_verdict_token("refused-correctly") == "refused-correctly"
    assert parse_verdict_token("refused") == "refused-correctly"
    assert parse_verdict_token("nope") is None
    assert parse_verdict_token("") is None


# ── join_grades ───────────────────────────────────────────────────────────


def test_join_histogram_math() -> None:
    rows = [_row("a", grounded=True), _row("b", grounded=False)]
    grades = [
        GradeRecord(case_id="a", investigation_id="inv-1",
                    verdict="supported", graded_at="t"),
        GradeRecord(case_id="b", investigation_id="inv-2",
                    verdict="unsupported", graded_at="t"),
    ]
    join = join_grades(rows, grades)
    assert isinstance(join, GradeJoin)
    assert join.histogram == {
        "supported": 1, "partial": 0, "unsupported": 1, "refused-correctly": 0,
    }
    assert join.graded == 2
    assert join.total == 2
    assert join.mean_grounded_ratio == pytest.approx(0.5)


def test_join_accepts_plain_dicts_and_prefers_recorded_features() -> None:
    rows = [dict(_row("a"), features={"grounded_ratio": 0.25})]
    join = join_grades(rows, [{"case_id": "a", "investigation_id": "inv-1",
                               "verdict": "partial", "graded_at": "t"}])
    assert join.histogram["partial"] == 1
    assert join.mean_grounded_ratio == pytest.approx(0.25)


def test_join_ignores_unknown_grades_and_reports_ungraded() -> None:
    rows = [_row("a")]
    join = join_grades(rows, [{"case_id": "ghost", "investigation_id": "inv-x",
                               "verdict": "supported", "graded_at": "t"}])
    assert join.histogram["supported"] == 0
    assert join.graded == 0
    assert join.total == 1
    assert join.mean_grounded_ratio == 0.0


def test_join_last_grade_wins() -> None:
    rows = [_row("a")]
    join = join_grades(rows, [
        {"case_id": "a", "investigation_id": "inv-1", "verdict": "supported", "graded_at": "t"},
        {"case_id": "a", "investigation_id": "inv-1", "verdict": "partial", "graded_at": "t2"},
    ])
    assert join.histogram == {
        "supported": 0, "partial": 1, "unsupported": 0, "refused-correctly": 0,
    }
    assert join.graded == 1


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_non_interactive_appends_and_resumes(tmp_path: Path) -> None:
    run_path = tmp_path / "20260908T120000Z.jsonl"
    _write_run(run_path, [_row("a"), _row("b")])
    grades_path = tmp_path / "20260908T120000Z.grades.jsonl"
    assert eval_grade.main(["grade", "--run", str(run_path),
                            "--non-interactive", "--verdict", "supported"]) == 0
    lines = grades_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["case_id"] == "a"
    assert first["verdict"] == "supported"
    assert first["investigation_id"] == "inv-1"
    assert first["graded_at"]
    # Second run with a different verdict changes nothing: graded rows skip.
    assert eval_grade.main(["--run", str(run_path),
                            "--non-interactive", "--verdict", "unsupported"]) == 0
    rerun = grades_path.read_text(encoding="utf-8").splitlines()
    assert len(rerun) == 2
    assert all(json.loads(line)["verdict"] == "supported" for line in rerun)


def test_cli_only_subset(tmp_path: Path) -> None:
    run_path = tmp_path / "run.jsonl"
    _write_run(run_path, [_row("a"), _row("b")])
    assert eval_grade.main(["grade", "--run", str(run_path), "--only", "a",
                            "--non-interactive", "--verdict", "partial",
                            "--notes", "half cited"]) == 0
    grades_path = tmp_path / "run.grades.jsonl"
    lines = grades_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["case_id"] == "a"
    assert record["notes"] == "half cited"


def test_cli_usage_errors(tmp_path: Path) -> None:
    assert eval_grade.main([]) == 2
    assert eval_grade.main(["grade"]) == 2
    assert eval_grade.main(["grade", "--run", str(tmp_path / "missing.jsonl"),
                            "--non-interactive", "--verdict", "supported"]) == 2
    run_path = tmp_path / "run.jsonl"
    _write_run(run_path, [_row("a")])
    # --non-interactive without --verdict, and --verdict without the flag.
    assert eval_grade.main(["grade", "--run", str(run_path), "--non-interactive"]) == 2
    assert eval_grade.main(["grade", "--run", str(run_path), "--verdict", "supported"]) == 2
    # Unknown case id.
    assert eval_grade.main(["grade", "--run", str(run_path), "--only", "ghost",
                            "--non-interactive", "--verdict", "supported"]) == 2
    # Corrupt run JSON.
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert eval_grade.main(["grade", "--run", str(bad),
                            "--non-interactive", "--verdict", "supported"]) == 2


def test_grades_path_for_sibling() -> None:
    assert (eval_grade.grades_path_for(Path("runs/20260908T120000Z.jsonl")).name
            == "20260908T120000Z.grades.jsonl")


def test_rubric_doc_exists() -> None:
    readme = Path("evals/README.md")
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8").lower()
    assert "rubric" in text
    for verdict in GRADE_VERDICTS:
        assert verdict in text
