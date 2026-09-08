"""P5-2 human grading models + offline join (DEC-054).

No LLM judge in this stage. A person reads each run row and assigns one of
four verdicts; this module holds the record shape and the join that turns a
run plus its grades file into a small histogram. Offline truth stays in
``runs/<ts>.grades.jsonl``; Grafana never sees per-question labels.

Rubric (full text in evals/README.md):
  supported: every claim grounded with at least one cited evidence id
    present on the board, and no invented sources.
  partial: at least half the claims grounded.
  unsupported: invented sources, or mostly ungrounded claims.
  refused-correctly: terminal in any state with zero fabrication and the
    limits stated (empty board, or only proposed claims citing nothing).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.eval.normalize import board_summary

GradeVerdict = Literal["supported", "partial", "unsupported", "refused-correctly"]

GRADE_VERDICTS: tuple[GradeVerdict, ...] = (
    "supported",
    "partial",
    "unsupported",
    "refused-correctly",
)

# Single-letter stdin shortcuts used by scripts/eval_grade.py.
VERDICT_SHORTCUTS: dict[str, GradeVerdict] = {
    "s": "supported",
    "p": "partial",
    "u": "unsupported",
    "r": "refused-correctly",
}

# One-line criteria shown by the grading CLI before each verdict prompt.
VERDICT_HELP: dict[str, str] = {
    "supported": "all claims grounded + cited, nothing invented",
    "partial": "at least half the claims grounded",
    "unsupported": "invented sources or mostly ungrounded",
    "refused-correctly": "any terminal state, zero fabrication, limits stated",
}


class GradeRecord(BaseModel):
    """One human judgment on a single run row."""

    case_id: str = Field(min_length=1)
    investigation_id: str = ""
    verdict: GradeVerdict
    notes: str = ""
    graded_at: str = Field(min_length=1)


class GradeJoin(BaseModel):
    """Result of joining candidate rows with their grades."""

    histogram: dict[str, int] = Field(default_factory=dict)
    mean_grounded_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    graded: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)


def parse_verdict_token(token: str) -> GradeVerdict | None:
    """Map a typed token to a verdict, or None when unrecognized.

    Accepts single letters (s/p/u/r) and full verdict names,
    case-insensitive, plus "refused" as an alias for refused-correctly.
    """
    key = token.strip().lower()
    if key in VERDICT_SHORTCUTS:
        return VERDICT_SHORTCUTS[key]
    for verdict in GRADE_VERDICTS:
        if key == verdict:
            return verdict
    if key == "refused":
        return "refused-correctly"
    return None


def _coerce_grade(raw: GradeRecord | Mapping[str, Any]) -> GradeRecord:
    if isinstance(raw, GradeRecord):
        return raw
    if isinstance(raw, Mapping):
        return GradeRecord.model_validate(dict(raw))
    raise TypeError(f"cannot join grade from {type(raw).__name__}")


def _row_grounded_ratio(row: Mapping[str, Any]) -> float:
    """Grounded ratio for one run row.

    Prefers the ``features`` aggregate recorded at run time; falls back to
    recomputing from the board so hand-written rows still join.
    """
    features = row.get("features")
    if isinstance(features, Mapping):
        value = features.get("grounded_ratio")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return min(max(float(value), 0.0), 1.0)
    return float(board_summary(dict(row))["grounded_ratio"])


def join_grades(
    candidate_rows: Sequence[Mapping[str, Any]],
    grades: Sequence[GradeRecord | Mapping[str, Any]],
) -> GradeJoin:
    """Join run rows with grades into a verdict histogram.

    Matching is by ``case_id``. Grades for unknown case ids are ignored, and
    when a case has several grades the last one wins. ``mean_grounded_ratio``
    averages the grounded ratio of the matched rows only; it is 0.0 when
    nothing has been graded yet.
    """
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id:
            rows_by_id[case_id] = row
    grades_by_id: dict[str, GradeRecord] = {}
    for raw in grades:
        grade = _coerce_grade(raw)
        grades_by_id[grade.case_id] = grade
    histogram: dict[str, int] = {verdict: 0 for verdict in GRADE_VERDICTS}
    ratios: list[float] = []
    matched = 0
    for case_id, grade in grades_by_id.items():
        matched_row = rows_by_id.get(case_id)
        if matched_row is None:
            continue
        histogram[grade.verdict] += 1
        ratios.append(_row_grounded_ratio(matched_row))
        matched += 1
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return GradeJoin(
        histogram=histogram,
        mean_grounded_ratio=mean_ratio,
        graded=matched,
        total=len(candidate_rows),
    )
