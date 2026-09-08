"""P5-1 baseline-diff regression gate CLI (DEC-054).

Compares a candidate run (live-eval JSONL rows or a baseline-shaped JSON
file) against a versioned baseline using aggregate features only. Exact
Markdown matching is forbidden anywhere in this pipeline.

Default gates (all must hold):
  verdict-drop>0 fails | grounding drop >5pp fails | p90 final-report +>25% fails.

Exit codes: 0 gate passed, 1 regression detected, 2 usage error.

Usage:
    python scripts/eval_diff.py --candidate runs/20260908T120000Z.jsonl
    python scripts/eval_diff.py --baseline evals/baselines/v0.6.0.json --candidate runs/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.diff import (  # noqa: E402
    DEFAULT_MAX_GROUNDING_DROP_PP,
    DEFAULT_MAX_P90_DELTA_PCT,
    CaseEntry,
    EvalBaseline,
    diff_runs,
)
from scripts.eval_live import validate_live_row  # noqa: E402

DEFAULT_BASELINE = "evals/baselines/v0.6.0.json"
GATES = ("verdict-drop", "grounding-drop", "p90-regression")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline-diff regression gate.")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", default=None, help="JSONL rows or baseline-shaped JSON.")
    parser.add_argument(
        "--fail-on", default=",".join(GATES),
        help=f"Comma-separated gates to enforce (default: all of {','.join(GATES)}).",
    )
    parser.add_argument("--max-grounding-drop-pp", type=float,
                        default=DEFAULT_MAX_GROUNDING_DROP_PP)
    parser.add_argument("--max-p90-delta-pct", type=float, default=DEFAULT_MAX_P90_DELTA_PCT)
    return parser


def _entry_from_row(row: dict[str, Any]) -> tuple[str, CaseEntry]:
    """Recompute features + failures from a raw row (single source of truth)."""
    features, failures = validate_live_row(row)
    elapsed = row.get("elapsed_s")
    return str(row.get("case_id", "unknown")), CaseEntry(
        terminal=features["terminal"],
        claim_count=int(features["claim_count"]),
        grounded_ratio=float(features["grounded_ratio"]),
        has_final_synthesis=bool(features["has_final_synthesis"]),
        refused_correctly=bool(features["refused_correctly"]),
        elapsed_s=float(elapsed) if isinstance(elapsed, (int, float)) else None,
        failures=list(failures),
    )


def load_candidate(path: Path) -> EvalBaseline:
    """Load JSONL live rows or a baseline-shaped JSON file as a baseline."""
    if path.suffix == ".jsonl":
        cases: dict[str, CaseEntry] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read candidate {path}: {exc}") from exc
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"candidate {path} line {lineno}: bad JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"candidate {path} line {lineno}: expected a JSON object")
            case_id, entry = _entry_from_row(row)
            cases[case_id] = entry
        return EvalBaseline(version=f"candidate:{path.name}", cases=cases)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read candidate {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidate {path}: bad JSON: {exc}") from exc
    return EvalBaseline.model_validate(raw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidate:
        print("Missing --candidate (JSONL rows or baseline-shaped JSON).", file=sys.stderr)
        return 2
    gates = [gate.strip() for gate in str(args.fail_on).split(",") if gate.strip()]
    unknown = sorted(set(gates) - set(GATES))
    if unknown:
        print(f"Unknown gates: {', '.join(unknown)} (choose from {', '.join(GATES)})",
              file=sys.stderr)
        return 2
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    try:
        baseline = EvalBaseline.model_validate(json.loads(baseline_path.read_text("utf-8")))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Bad baseline {baseline_path}: {exc}", file=sys.stderr)
        return 2
    try:
        candidate = load_candidate(candidate_path)
    except ValueError as exc:
        print(f"Bad candidate: {exc}", file=sys.stderr)
        return 2

    diff = diff_runs(
        baseline, candidate,
        max_grounding_drop_pp=args.max_grounding_drop_pp,
        max_p90_delta_pct=args.max_p90_delta_pct,
    )
    reasons: list[str] = []
    if "verdict-drop" in gates and diff.verdict_drops > 0:
        reasons.append(f"verdict-drops={diff.verdict_drops} (>0)")
    if "grounding-drop" in gates and diff.grounding_delta_pp < -args.max_grounding_drop_pp:
        reasons.append(
            f"grounding-delta={diff.grounding_delta_pp:+.2f}pp "
            f"(< -{args.max_grounding_drop_pp:.2f}pp)"
        )
    if "p90-regression" in gates and diff.p90_final_report_delta_pct > args.max_p90_delta_pct:
        reasons.append(
            f"p90-final-report-delta={diff.p90_final_report_delta_pct:+.1f}% "
            f"(> +{args.max_p90_delta_pct:.1f}%)"
        )

    print(f"compared={diff.compared_cases} "
          f"(baseline={diff.baseline_cases}, candidate={diff.candidate_cases})")
    print(f"verdict_drops={diff.verdict_drops} "
          f"grounding_delta_pp={diff.grounding_delta_pp:+.2f} "
          f"p90_final_report_delta_pct={diff.p90_final_report_delta_pct:+.1f}")
    if reasons:
        for reason in reasons:
            print(f"FAIL: {reason}", file=sys.stderr)
        print(f"Diff FAILED on {len(reasons)} gate(s).", file=sys.stderr)
        return 1
    print("Diff OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
