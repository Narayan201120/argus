"""P5-1 baseline-diff regression gate (DEC-054).

Compares aggregate run features only. Exact Markdown matching is forbidden:
reports never enter the diff except through normalized aggregates recorded
at run time (grounded_ratio, timings, pass/fail).

Verdict semantics (DEC-054): a "verdict drop" means a case that passed
validation in the baseline run fails validation in the candidate run.
Verdicts never constrain terminal state; any terminal state can pass when
the board shows zero fabrication.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

# Default regression thresholds (see scripts/eval_diff.py for CLI overrides).
DEFAULT_MAX_GROUNDING_DROP_PP = 5.0
DEFAULT_MAX_P90_DELTA_PCT = 25.0


class CaseEntry(BaseModel):
    """Per-case aggregate recorded by a live eval run."""

    terminal: str | None = None
    claim_count: int = Field(default=0, ge=0)
    grounded_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    has_final_synthesis: bool = False
    refused_correctly: bool = False
    elapsed_s: float | None = Field(default=None, ge=0.0)
    failures: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


class EvalBaseline(BaseModel):
    """Baseline/candidate file shape. Extra keys (``_schema``, ``_note``) ignored."""

    model_config = {"extra": "ignore"}

    version: str = "unknown"
    cases: dict[str, CaseEntry] = Field(default_factory=dict)


class EvalDiff(BaseModel):
    verdict_drops: int = Field(ge=0)
    grounding_delta_pp: float
    p90_final_report_delta_pct: float
    passed: bool
    compared_cases: int = Field(ge=0)
    baseline_cases: int = Field(ge=0)
    candidate_cases: int = Field(ge=0)


def _coerce_cases(source: EvalBaseline | Mapping[str, Any]) -> dict[str, CaseEntry]:
    if isinstance(source, EvalBaseline):
        return dict(source.cases)
    if isinstance(source, Mapping):
        raw = source.get("cases", {})
        entries: dict[str, CaseEntry] = {}
        if isinstance(raw, Mapping):
            for case_id, item in raw.items():
                if isinstance(item, CaseEntry):
                    entries[str(case_id)] = item
                elif isinstance(item, Mapping):
                    entries[str(case_id)] = CaseEntry.model_validate(dict(item))
        return entries
    raise TypeError(f"cannot diff from {type(source).__name__}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def p90(values: list[float]) -> float | None:
    """Nearest-rank 90th percentile; None when there is no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(0.9 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _final_timings(cases: Mapping[str, CaseEntry]) -> list[float]:
    return [
        entry.elapsed_s
        for entry in cases.values()
        if entry.has_final_synthesis and entry.elapsed_s is not None
    ]


def diff_runs(
    baseline: EvalBaseline | Mapping[str, Any],
    candidate: EvalBaseline | Mapping[str, Any],
    max_grounding_drop_pp: float = DEFAULT_MAX_GROUNDING_DROP_PP,
    max_p90_delta_pct: float = DEFAULT_MAX_P90_DELTA_PCT,
) -> EvalDiff:
    """Diff two runs over their shared cases.

    Empty intersections (including the first-live-run empty baseline) yield
    zero deltas and ``passed=True``: there is nothing to regress against.
    """
    base = _coerce_cases(baseline)
    cand = _coerce_cases(candidate)
    compared = sorted(set(base) & set(cand))

    verdict_drops = sum(1 for cid in compared if base[cid].passed and not cand[cid].passed)

    base_grounding = [base[cid].grounded_ratio for cid in compared]
    cand_grounding = [cand[cid].grounded_ratio for cid in compared]
    grounding_delta_pp = (_mean(cand_grounding) - _mean(base_grounding)) * 100.0 if compared else 0.0

    base_p90 = p90(_final_timings({cid: base[cid] for cid in compared}))
    cand_p90 = p90(_final_timings({cid: cand[cid] for cid in compared}))
    if base_p90 is None or cand_p90 is None or base_p90 <= 0:
        p90_delta_pct = 0.0
    else:
        p90_delta_pct = (cand_p90 - base_p90) / base_p90 * 100.0

    passed = (
        verdict_drops == 0
        and grounding_delta_pp >= -max_grounding_drop_pp
        and p90_delta_pct <= max_p90_delta_pct
    )
    return EvalDiff(
        verdict_drops=verdict_drops,
        grounding_delta_pp=grounding_delta_pp,
        p90_final_report_delta_pct=p90_delta_pct,
        passed=passed,
        compared_cases=len(compared),
        baseline_cases=len(base),
        candidate_cases=len(cand),
    )
