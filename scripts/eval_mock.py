"""P5-0 mock-only offline eval runner (DEC-054).

Builds canned boards per corpus case entirely in-process and runs the pure
validators over them. There is deliberately NO --live flag: this script
cannot spend quota by construction (no httpx, no provider SDK imports).

Exit codes: 0 all cases pass, 1 any case fails, 2 bad corpus/args.

Usage:
    python scripts/eval_mock.py
    python scripts/eval_mock.py --only radar-01 rag-01
    python scripts/eval_mock.py --list
    python scripts/eval_mock.py --sabotage drop-citations   # must exit 1
    python scripts/eval_mock.py --sabotage empty-board       # must exit 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.corpus import EvalCase, load_corpus
from app.eval.validators import (
    validate_board_shape,
    validate_claim_grounding,
    validate_honest_gap,
)

DEFAULT_CORPUS = "evals/corpus.yaml"
SABOTAGE_MODES = ("drop-citations", "empty-board")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock-only offline eval runner.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--only", nargs="*", default=None, help="Subset of case ids to run.")
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument(
        "--sabotage",
        choices=SABOTAGE_MODES,
        default=None,
        help="Corrupt canned boards to prove validators catch fabrication.",
    )
    return parser


def build_board(case: EvalCase) -> tuple[dict[str, Any], list[Any]]:
    """Construct a canned board plus syntheses for *case*.

    Answerable buckets get a well-formed board with grounded claims;
    refused-correctly cases get an empty board with zero claims and a
    refusal synthesis (zero fabrication, DEC-054).
    """
    if case.expect.verdict == "refused-correctly":
        board: dict[str, Any] = {"evidence": [], "claims": []}
        syntheses: list[Any] = [{"markdown": "I cannot answer this.", "source_refs": []}]
        return board, syntheses
    count = max(case.expect.min_evidence, 1)
    evidence = [
        {
            "id": f"{case.id}-ev-{index}",
            "investigation_id": case.id,
            "source_ref": f"{case.id}-ref-{index}",
            "content": f"canned mock finding {index} for {case.query}",
            "type": "text",
            "confidence": 0.8,
            "created_at": 1700000000.0 + index,
        }
        for index in range(count)
    ]
    cited = [item["id"] for item in evidence]
    claims = [
        {
            "id": f"{case.id}-cl-0",
            "investigation_id": case.id,
            "statement": f"canned mock answer to: {case.query}",
            "confidence": 0.8,
            "evidence_ids": cited,
            "status": "supported",
        }
    ]
    refs = [item["source_ref"] for item in evidence]
    syntheses = [
        {
            "markdown": "canned mock synthesis " + " ".join(f"source={ref}" for ref in refs),
            "source_refs": list(refs),
        }
    ]
    return {"evidence": evidence, "claims": claims}, syntheses


def apply_sabotage(
    board: dict[str, Any], syntheses: list[Any], mode: str | None
) -> tuple[dict[str, Any], list[Any]]:
    if mode == "drop-citations":
        for claim in board["claims"]:
            claim["evidence_ids"] = []
    elif mode == "empty-board":
        board["evidence"] = []
    return board, syntheses


def check_case(case: EvalCase, sabotage: str | None) -> list[str]:
    board, syntheses = apply_sabotage(*build_board(case), sabotage)
    failures = [f"shape: {e}" for e in validate_board_shape(board)]
    ratio, grounding_errors = validate_claim_grounding(board)
    failures += [f"grounding: {e}" for e in grounding_errors]
    failures += [f"honest-gap: {e}" for e in validate_honest_gap(board, syntheses)]
    evidence = board["evidence"] if isinstance(board.get("evidence"), list) else []
    claims = board["claims"] if isinstance(board.get("claims"), list) else []
    if len(evidence) < case.expect.min_evidence:
        failures.append(
            f"expectation: {len(evidence)} evidence < min_evidence {case.expect.min_evidence}"
        )
    if case.expect.verdict == "refused-correctly":
        if claims:
            failures.append(f"expectation: refused-correctly has {len(claims)} claim(s), want zero")
        if evidence:
            failures.append(
                f"expectation: refused-correctly has {len(evidence)} evidence, want zero"
            )
    if case.expect.must_cite and claims:
        for claim in claims:
            cited = claim.get("evidence_ids")
            if not isinstance(cited, list) or not cited:
                failures.append(f"expectation: claim {claim.get('id', '?')!r} cites nothing")
    print(f"  grounded_ratio={ratio:.2f} evidence={len(evidence)} claims={len(claims)}")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_corpus(args.corpus)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Bad corpus: {exc}", file=sys.stderr)
        return 2
    if args.list:
        for case in cases:
            print(f"{case.id} {case.bucket} {case.expect.verdict}")
        return 0
    selected = cases
    if args.only is not None:
        wanted = set(args.only)
        known = {case.id for case in cases}
        unknown = sorted(wanted - known)
        if unknown:
            print(f"Unknown case ids: {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected = [case for case in cases if case.id in wanted]
        if not selected:
            print("No cases selected.", file=sys.stderr)
            return 2
    failures = 0
    for case in selected:
        errors = check_case(case, args.sabotage)
        status = "PASS" if not errors else "FAIL"
        if errors:
            failures += 1
        print(f"[{status}] {case.id} ({case.bucket}, expect {case.expect.verdict})")
        for error in errors:
            print(f"    - {error}")
    sabotage_note = f" sabotage={args.sabotage}" if args.sabotage else ""
    print(f"Eval{sabotage_note}: {len(selected) - failures}/{len(selected)} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
