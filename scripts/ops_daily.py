"""P6-2 daily ops probe (scripts + docs + tests only; no server changes).

Runs the fixed 5-query set against a live ARGUS server through
POST /v1/investigate, polls each row to a terminal state, and appends one
JSONL row per query to runs/ops-YYYY-MM-DD.jsonl.

Fixed query set:
  radar     "dense vs sparse retrieval scientific QA"       (Radar-heavy)
  rag       "ARGUS response-cache TTL policy"               (RAG-heavy)
  web       "latest Tavily API pricing"                     (web-heavy, needs-web)
  poisoned  "Prove P=NP using our library"                  (expect fast FAILED)
  dupe      "large language models industrial applications" (dupe-prone, expect COMPLETE)

Per-kind expectations (strict and explicit):
  radar/rag/web: COMPLETE, or BUDGET_EXHAUSTED with a documented budget
    reason (cost_limit, iteration_limit, tool_call_limit, wall_clock_limit).
  poisoned:      FAILED with reason provider_failure within 60 s.
  dupe:          COMPLETE (budget states do not count here).

The web query is skipped gracefully (verdict "skipped", never a failure)
when web tools are off; see probe_web_tools().

Usage:
    python scripts/ops_daily.py --dry-run
    python scripts/ops_daily.py --live
    ARGUS_OPS_LIVE=1 python scripts/ops_daily.py --only radar poisoned

Exit codes: 0 all-as-expected (skips ok), 1 any mismatch/infra row,
2 refused to run (guard or spend-cap abort), 3 usage/infra error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from scripts.eval_live import estimate_per_run_usd  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_POLL_S = 5.0
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_SPEND_CAP_USD = 2.00

TERMINAL_STATUSES = {"complete", "failed", "budget_exhausted", "cancelled"}
DOCUMENTED_BUDGET_REASONS = {"cost_limit", "iteration_limit", "tool_call_limit", "wall_clock_limit"}
POISON_MAX_ELAPSED_S = 60.0

DAILY_QUERIES: list[dict[str, str]] = [
    {"kind": "radar", "query": "dense vs sparse retrieval scientific QA"},
    {"kind": "rag", "query": "ARGUS response-cache TTL policy"},
    {"kind": "web", "query": "latest Tavily API pricing"},
    {"kind": "poisoned", "query": "Prove P=NP using our library"},
    {"kind": "dupe", "query": "large language models industrial applications"},
]
KINDS = tuple(item["kind"] for item in DAILY_QUERIES)

VERDICTS_OK = {"as-expected", "skipped"}


class OpsInfraError(RuntimeError):
    """Transport, HTTP-status, or protocol failure talking to the server."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


# ── Pure helpers (unit-testable without network) ─────────────────────────────


def check_expectations(
    kind: str, *, status: Any, status_reason: Any, elapsed_s: float
) -> tuple[str, str]:
    """Compare one terminal snapshot against the per-kind expectation.

    Returns (verdict, note) with verdict in {"as-expected", "mismatch"}.
    """
    if kind == "poisoned":
        if status != "failed":
            return "mismatch", f"poisoned: expected FAILED, got {status!r}"
        if status_reason != "provider_failure":
            return "mismatch", f"poisoned: expected reason provider_failure, got {status_reason!r}"
        if elapsed_s > POISON_MAX_ELAPSED_S:
            return (
                "mismatch",
                f"poisoned: FAILED after {elapsed_s:.1f}s (limit {POISON_MAX_ELAPSED_S:.0f}s)",
            )
        return "as-expected", f"fast FAILED/provider_failure in {elapsed_s:.1f}s"
    if kind == "dupe":
        if status == "complete":
            return "as-expected", f"COMPLETE/{status_reason} in {elapsed_s:.1f}s"
        return "mismatch", f"dupe: expected COMPLETE, got {status!r}/{status_reason!r}"
    if status == "complete":
        return "as-expected", f"COMPLETE/{status_reason} in {elapsed_s:.1f}s"
    if status == "budget_exhausted" and status_reason in DOCUMENTED_BUDGET_REASONS:
        return "as-expected", f"documented-budget {status}/{status_reason} in {elapsed_s:.1f}s"
    return (
        "mismatch",
        f"{kind}: expected COMPLETE (or documented-budget), got {status!r}/{status_reason!r}",
    )


def probe_web_tools(client: httpx.Client) -> tuple[bool, str]:
    """Decide whether to run the web-heavy query. Never raises.

    Skip signals: ARGUS_OPS_SKIP_WEB=1, or workspace.web=false on
    GET /v1/meta. Absent any off-signal the query runs.
    """
    if os.environ.get("ARGUS_OPS_SKIP_WEB", "").lower() in {"1", "true", "yes"}:
        return False, "ARGUS_OPS_SKIP_WEB set"
    try:
        response = client.get("/v1/meta", timeout=15.0)
    except httpx.HTTPError as exc:
        return True, f"meta probe failed ({exc}); attempting web query anyway"
    if response.status_code != 200:
        return True, f"meta HTTP {response.status_code}; attempting web query anyway"
    try:
        payload = response.json()
    except ValueError:
        return True, "meta returned non-JSON; attempting web query anyway"
    workspace = payload.get("workspace") if isinstance(payload, dict) else None
    if isinstance(workspace, dict) and workspace.get("web") is False:
        return False, "server meta reports workspace.web=false"
    return True, "no web-off signal; attempting web query"


# ── Network runner ───────────────────────────────────────────────────────────


def run_one(client: httpx.Client, query: str, *, poll_s: float, timeout_s: float) -> dict[str, Any]:
    """POST one query, poll the board to terminal, return the snapshot."""
    started = time.monotonic()
    try:
        created = client.post("/v1/investigate", json={"query": query}, timeout=30.0)
    except httpx.HTTPError as exc:
        raise OpsInfraError(f"investigate POST failed: {exc}") from exc
    if created.status_code != 202:
        raise OpsInfraError(f"investigate POST HTTP {created.status_code}", created.status_code)
    try:
        payload = created.json()
    except ValueError as exc:
        raise OpsInfraError("investigate POST returned non-JSON") from exc
    investigation_id = payload.get("investigation_id") if isinstance(payload, dict) else None
    if not investigation_id:
        raise OpsInfraError("investigate POST returned no investigation_id")

    poll_url = f"/v1/investigate/{investigation_id}"
    body: dict[str, Any] = {}
    timed_out = False
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            response = client.get(poll_url, timeout=15.0)
        except httpx.HTTPError as exc:
            raise OpsInfraError(f"board poll failed: {exc}") from exc
        if response.status_code != 200:
            raise OpsInfraError(f"board poll HTTP {response.status_code}", response.status_code)
        try:
            data = response.json()
        except ValueError as exc:
            raise OpsInfraError("board poll returned non-JSON") from exc
        body = data if isinstance(data, dict) else {}
        if body.get("status") in TERMINAL_STATUSES:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        time.sleep(min(poll_s, remaining))
    elapsed_s = time.monotonic() - started
    evidence = body.get("evidence")
    claims = body.get("claims")
    return {
        "investigation_id": investigation_id,
        "status": body.get("status"),
        "status_reason": body.get("status_reason"),
        "evidence": len(evidence) if isinstance(evidence, list) else 0,
        "claims": len(claims) if isinstance(claims, list) else 0,
        "elapsed_s": elapsed_s,
        "timed_out": timed_out,
        "created_at": body.get("created_at"),
        "updated_at": body.get("updated_at"),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily ARGUS ops probe (fixed 5-query set).")
    parser.add_argument("--base-url", default=os.environ.get("ARGUS_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--only", nargs="*", default=None, help=f"Subset of kinds {list(KINDS)}.")
    parser.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--spend-cap-usd",
        type=float,
        default=DEFAULT_SPEND_CAP_USD,
        help="Abort (exit 2) when the pre-flight ceiling exceeds this.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to actually run; alternatively set ARGUS_OPS_LIVE=1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned POSTs + spend ceiling, run nothing, exit 0.",
    )
    parser.add_argument("--out", default=None, help="JSONL output path (default runs/ops-YYYY-MM-DD.jsonl).")
    return parser


def _default_out() -> str:
    return f"runs/ops-{datetime.now().date().isoformat()}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wanted = list(args.only) if args.only else list(KINDS)
    unknown = [kind for kind in wanted if kind not in KINDS]
    if unknown:
        print(f"Unknown kinds: {', '.join(unknown)} (valid: {', '.join(KINDS)})", file=sys.stderr)
        return 3
    selected = [item for item in DAILY_QUERIES if item["kind"] in wanted]

    per_run = estimate_per_run_usd()
    ceiling = len(selected) * per_run
    if args.dry_run:
        print(
            f"Dry run: {len(selected)} investigate POST(s) planned "
            f"(kinds: {', '.join(wanted)}), spend ceiling estimate ${ceiling:.2f} "
            f"(${per_run:.2f}/run x {len(selected)}, cap ${args.spend_cap_usd:.2f})."
        )
        for item in selected:
            print(f"  POST /v1/investigate kind={item['kind']} query={item['query']!r}")
        if ceiling > args.spend_cap_usd:
            print("Note: a live run with this plan would abort (ceiling over cap).")
        return 0

    unlocked = args.live or os.environ.get("ARGUS_OPS_LIVE", "").lower() in {"1", "true"}
    if not unlocked:
        print(
            "Refusing to run: daily ops spends real provider tokens.\n"
            "Pass --live or set ARGUS_OPS_LIVE=1.",
            file=sys.stderr,
        )
        return 2
    if ceiling > args.spend_cap_usd:
        print(
            f"Refusing to run: spend ceiling estimate ${ceiling:.2f} exceeds "
            f"cap ${args.spend_cap_usd:.2f} "
            f"(${per_run:.2f}/run x {len(selected)}). "
            "Drop kinds with --only or raise --spend-cap-usd.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out) if args.out else Path(_default_out())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date().isoformat()
    cost_note = (
        f"client-side estimate ~${per_run:.2f}/run; server-side cost_usd is "
        "authoritative, real billing lives in provider consoles"
    )

    try:
        client = httpx.Client(base_url=args.base_url, timeout=45.0)
    except Exception as exc:  # noqa: BLE001 - client construction must not traceback
        print(f"Cannot create HTTP client: {exc}", file=sys.stderr)
        return 3

    problems = 0
    with client:
        run_web, web_reason = probe_web_tools(client)
        if "web" in wanted:
            print(f"web-tools probe: run_web={run_web} ({web_reason})")
        with out_path.open("a", encoding="utf-8") as handle:
            for item in selected:
                kind = item["kind"]
                if kind == "web" and not run_web:
                    row: dict[str, Any] = {
                        "date": today,
                        "query": item["query"],
                        "kind": kind,
                        "investigation_id": None,
                        "status": "skipped",
                        "status_reason": None,
                        "evidence": 0,
                        "claims": 0,
                        "elapsed_s": 0.0,
                        "timed_out": False,
                        "verdict": "skipped",
                        "expected": "COMPLETE-or-skip",
                        "note": f"skipped: {web_reason}",
                        "cost_note": cost_note,
                        "created_at": None,
                        "updated_at": None,
                        "http_status": None,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(f"[SKIP] {kind} ({web_reason})")
                    continue
                if kind == "poisoned":
                    expected = "FAILED/provider_failure<=60s"
                else:
                    expected = "COMPLETE-or-documented"
                try:
                    snap = run_one(client, item["query"], poll_s=args.poll_s, timeout_s=args.timeout_s)
                except OpsInfraError as exc:
                    row = {
                        "date": today,
                        "query": item["query"],
                        "kind": kind,
                        "investigation_id": None,
                        "status": "infra-error",
                        "status_reason": None,
                        "evidence": 0,
                        "claims": 0,
                        "elapsed_s": 0.0,
                        "timed_out": False,
                        "verdict": "infra-error",
                        "expected": expected,
                        "note": f"infra-error: {exc}",
                        "cost_note": cost_note,
                        "created_at": None,
                        "updated_at": None,
                        "http_status": exc.http_status,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    problems += 1
                    print(f"[INFRA] {kind} {exc}", file=sys.stderr)
                    continue
                verdict, note = check_expectations(
                    kind,
                    status=snap["status"],
                    status_reason=snap["status_reason"],
                    elapsed_s=snap["elapsed_s"],
                )
                row = {
                    "date": today,
                    "query": item["query"],
                    "kind": kind,
                    "investigation_id": snap["investigation_id"],
                    "status": snap["status"],
                    "status_reason": snap["status_reason"],
                    "evidence": snap["evidence"],
                    "claims": snap["claims"],
                    "elapsed_s": snap["elapsed_s"],
                    "timed_out": snap["timed_out"],
                    "verdict": verdict,
                    "expected": expected,
                    "note": note,
                    "cost_note": cost_note,
                    "created_at": snap["created_at"],
                    "updated_at": snap["updated_at"],
                    "http_status": None,
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if verdict != "as-expected":
                    problems += 1
                tag = "ok" if verdict == "as-expected" else "MISMATCH"
                print(
                    f"[{tag}] {kind} status={snap['status']}/{snap['status_reason']} "
                    f"ev={snap['evidence']} cl={snap['claims']} {snap['elapsed_s']:.1f}s -- {note}"
                )

    ran = len(selected)
    print(f"Ops daily: {ran - problems}/{ran} as-expected-or-skipped (rows in {out_path}).")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
