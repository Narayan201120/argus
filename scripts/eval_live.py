"""P5-1 live eval runner (DEC-054).

Runs corpus cases against a live ARGUS server through POST /v1/investigate
and polls GET /v1/investigate/{id} to a terminal state, appending one JSONL
row per case. Verdicts never constrain terminal state: any terminal state
passes when the board shows zero fabrication (see app/eval/validators.py).

Per DEC-054 this script refuses to run unless explicitly unlocked with
--live or ARGUS_EVAL_LIVE=1, because live runs spend provider tokens.

Usage:
    python scripts/eval_live.py --dry-run
    python scripts/eval_live.py --live --max-runs 3
    ARGUS_EVAL_LIVE=1 python scripts/eval_live.py --only radar-01 rag-01

Exit codes: 0 all cases pass, 1 any case fails, 2 refused to run (guard or
spend-cap abort), 3 usage/infra error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import settings  # noqa: E402
from app.eval.corpus import EvalCase, load_corpus  # noqa: E402
from app.eval.normalize import board_summary  # noqa: E402
from app.eval.validators import validate_claim_grounding, validate_honest_gap  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_CORPUS = "evals/corpus.yaml"
DEFAULT_POLL_S = 10.0
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_SPEND_CAP_USD = 2.00

TERMINAL_STATUSES = {"complete", "failed", "budget_exhausted", "cancelled"}

# Spend-ceiling estimate inputs. Web cost comes from settings (max_web_calls
# x cost_usd_per_web_search); LLM cost is a ROUGH GUESS for loop + milestone
# synthesis calls per run, because per-run token totals are only known after
# the run. Real billing lives in provider consoles. The pre-flight ceiling
# (planned runs x per-run estimate) is deliberately pessimistic.
EST_LLM_USD_PER_RUN = 0.25


class EvalInfraError(RuntimeError):
    """Transport, HTTP-status, or protocol failure talking to the server."""


# ── Pure validators (unit-testable without network) ─────────────────────────


def validate_live_row(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate one live-run snapshot row.

    Returns (features, failures): features are the verdict-relevant
    aggregates from board_summary, failures is empty on pass. DEC-054: any
    terminal state passes with zero fabrication; only fabrication (dangling
    citations, supported-without-evidence, invented refs) or a missing final
    synthesis on ``complete`` fails.
    """
    failures: list[str] = []
    status = row.get("status")
    if status not in TERMINAL_STATUSES:
        failures.append(f"status={status!r} is not terminal (timed_out={row.get('timed_out')!r})")
    syntheses_raw = row.get("syntheses")
    syntheses: list[Any] = list(syntheses_raw) if isinstance(syntheses_raw, list) else []
    if status == "complete" and not any(
        isinstance(entry, dict) and entry.get("final") is True for entry in syntheses
    ):
        failures.append("status=complete without a final synthesis")
    claims_raw = row.get("claims")
    claims: list[Any] = list(claims_raw) if isinstance(claims_raw, list) else []
    evidence_raw = row.get("evidence_ids")
    evidence_ids: list[Any] = list(evidence_raw) if isinstance(evidence_raw, list) else []
    board = {
        "evidence": [{"id": eid} for eid in evidence_ids if isinstance(eid, str)],
        "claims": [claim for claim in claims if isinstance(claim, dict)],
    }
    _, grounding_errors = validate_claim_grounding(board)
    failures += [f"grounding: {error}" for error in grounding_errors]
    failures += [f"honest-gap: {error}" for error in validate_honest_gap(board, syntheses)]
    return board_summary(row), failures


def estimate_per_run_usd() -> float:
    """Pessimistic per-run spend ceiling from settings plus a documented guess."""
    return settings.max_web_calls * settings.cost_usd_per_web_search + EST_LLM_USD_PER_RUN


# ── Network runner ──────────────────────────────────────────────────────────


def run_one(
    client: httpx.Client, case: EvalCase, *, poll_s: float, timeout_s: float
) -> dict[str, Any]:
    """POST one case, poll to terminal, return the snapshot row. Sequential use."""
    started = time.monotonic()
    try:
        created = client.post("/v1/investigate", json={"query": case.query}, timeout=30.0)
    except httpx.HTTPError as exc:
        raise EvalInfraError(f"{case.id}: investigate POST failed: {exc}") from exc
    if created.status_code != 202:
        raise EvalInfraError(f"{case.id}: investigate POST HTTP {created.status_code}")
    try:
        investigation_id = created.json().get("investigation_id")
    except ValueError as exc:
        raise EvalInfraError(f"{case.id}: investigate POST returned non-JSON") from exc
    if not investigation_id:
        raise EvalInfraError(f"{case.id}: investigate POST returned no investigation_id")

    poll_url = f"/v1/investigate/{investigation_id}"
    body: dict[str, Any] = {}
    timed_out = False
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            response = client.get(poll_url, timeout=15.0)
        except httpx.HTTPError as exc:
            raise EvalInfraError(f"{case.id}: board poll failed: {exc}") from exc
        if response.status_code != 200:
            raise EvalInfraError(f"{case.id}: board poll HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise EvalInfraError(f"{case.id}: board poll returned non-JSON") from exc
        if body.get("status") in TERMINAL_STATUSES:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        time.sleep(min(poll_s, remaining))
    elapsed_s = time.monotonic() - started

    claims_raw = body.get("claims")
    claim_items: list[Any] = list(claims_raw) if isinstance(claims_raw, list) else []
    claims: list[dict[str, Any]] = []
    for index, raw in enumerate(claim_items):
        if not isinstance(raw, dict):
            continue
        claims.append(
            {
                "id": raw.get("id", f"claim-{index}"),
                "statement": raw.get("statement", ""),
                "confidence": raw.get("confidence", 0.0),
                "evidence_ids": raw.get("evidence_ids", []),
                "status": str(raw.get("status", "")),
            }
        )
    evidence_raw = body.get("evidence")
    evidence_items: list[Any] = list(evidence_raw) if isinstance(evidence_raw, list) else []
    evidence_ids = [item.get("id") for item in evidence_items if isinstance(item, dict)]
    syntheses_raw = body.get("syntheses")
    synthesis_items: list[Any] = list(syntheses_raw) if isinstance(syntheses_raw, list) else []
    syntheses = [
        {
            "milestone": entry.get("milestone", 0),
            "final": bool(entry.get("final", False)),
            "markdown": str(entry.get("markdown", "")),
        }
        for entry in synthesis_items
        if isinstance(entry, dict)
    ]
    final_markdown = next(
        (entry["markdown"] for entry in syntheses if entry["final"]), ""
    )
    return {
        "case_id": case.id,
        "query": case.query,
        "investigation_id": investigation_id,
        "status": body.get("status"),
        "status_reason": body.get("status_reason"),
        "counts": {"evidence": len(evidence_items), "claims": len(claims)},
        "evidence_ids": evidence_ids,
        "claims": claims,
        "syntheses": syntheses,
        "final_markdown": final_markdown,
        "elapsed_s": elapsed_s,
        "timed_out": timed_out,
        # InvestigationBoardResponse carries no token-usage field, so rows
        # record elapsed_s + terminal counts only; server-side cost_usd is
        # the authority and is not visible from this client.
        "usage_note": "no token usage on board GET; spend tracked via elapsed_s + counts",
    }


# ── CLI ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live eval runner for ARGUS.")
    parser.add_argument("--base-url", default=os.environ.get("ARGUS_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--only", nargs="*", default=None, help="Subset of case ids to run.")
    parser.add_argument("--max-runs", type=int, default=None, help="Cap on cases executed.")
    parser.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--spend-cap-usd", type=float, default=DEFAULT_SPEND_CAP_USD,
        help="Abort (exit 2) when the pre-flight ceiling exceeds this.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Required to actually run; alternatively set ARGUS_EVAL_LIVE=1.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned POST count + spend ceiling, run nothing, exit 0.",
    )
    parser.add_argument("--out", default=None, help="JSONL output path (default runs/<ts>.jsonl).")
    return parser


def _default_out() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"runs/{stamp}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_corpus(args.corpus)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Bad corpus: {exc}", file=sys.stderr)
        return 3
    selected = cases
    if args.only is not None:
        wanted = set(args.only)
        unknown = sorted(wanted - {case.id for case in cases})
        if unknown:
            print(f"Unknown case ids: {', '.join(unknown)}", file=sys.stderr)
            return 3
        selected = [case for case in cases if case.id in wanted]
    if args.max_runs is not None:
        if args.max_runs < 0:
            print(f"--max-runs must be >= 0, got {args.max_runs}", file=sys.stderr)
            return 3
        selected = selected[: args.max_runs]
    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 3

    per_run = estimate_per_run_usd()
    ceiling = len(selected) * per_run
    if args.dry_run:
        print(
            f"Dry run: {len(selected)} investigate POST(s) planned, "
            f"spend ceiling estimate ${ceiling:.2f} "
            f"(${per_run:.2f}/run x {len(selected)}, cap ${args.spend_cap_usd:.2f})."
        )
        if ceiling > args.spend_cap_usd:
            print("Note: a live run with this plan would abort (ceiling over cap).")
        return 0

    unlocked = args.live or os.environ.get("ARGUS_EVAL_LIVE", "").lower() in {"1", "true"}
    if not unlocked:
        print(
            "Refusing to run: live eval spends real provider tokens.\n"
            "Pass --live or set ARGUS_EVAL_LIVE=1 once the server has keys.",
            file=sys.stderr,
        )
        return 2
    if ceiling > args.spend_cap_usd:
        print(
            f"Refusing to run: spend ceiling estimate ${ceiling:.2f} exceeds "
            f"cap ${args.spend_cap_usd:.2f} "
            f"(${per_run:.2f}/run x {len(selected)}). "
            "Lower --max-runs/--only or raise --spend-cap-usd.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out) if args.out else Path(_default_out())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with httpx.Client(base_url=args.base_url, timeout=45.0) as client:
        with out_path.open("a", encoding="utf-8") as handle:
            for case in selected:
                try:
                    row = run_one(client, case, poll_s=args.poll_s, timeout_s=args.timeout_s)
                except EvalInfraError as exc:
                    print(f"FAIL: {exc}", file=sys.stderr)
                    print(f"Eval live FAILED on infra error; partial rows in {out_path}.",
                          file=sys.stderr)
                    return 3
                features, errors = validate_live_row(row)
                row["features"] = features
                row["failures"] = errors
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                status = "PASS" if not errors else "FAIL"
                if errors:
                    failures += 1
                print(f"[{status}] {case.id} terminal={row.get('status')} "
                      f"grounded={features['grounded_ratio']:.2f} "
                      f"elapsed={row['elapsed_s']:.1f}s")
                for error in errors:
                    print(f"    - {error}")

    print(f"Eval live: {len(selected) - failures}/{len(selected)} passed (rows in {out_path}).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
