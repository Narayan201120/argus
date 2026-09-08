"""P6-2 week-green checker (scripts + docs + tests only; no server changes).

Reads runs/ops-YYYY-MM-DD.jsonl (daily) and runs/ops-weekly-*.jsonl
(weekly) rows for the last 7 days and evaluates five gates:

  1. dailies:   >=23/25 weekday dailies as-expected. A missing weekday
                file is an automatic RED (stated, never tolerated).
                Rows with verdict "skipped" (e.g. web with WEB_TOOLS off)
                are excluded from the denominator and the bar becomes
                denominator-2, which equals 23/25 with no skips.
  2. orphans:   zero non-terminal rows past deadline+60 s (daily rows
                with timed_out or a live non-terminal status, plus
                orphans recorded by the weekly restart drill).
  3. p90:       p90 of elapsed_s over COMPLETE daily rows < 300 s
                (nearest-rank method).
  4. 500s:      zero rows with http_status 500.
  5. drills:    each of redis-drill/restart/smoke has >=1 pass and zero
                fails in the window. A skipped drill does not fail the
                gate but a drill with no passing row does (stated).

Usage:
    python scripts/green_week.py
    python scripts/green_week.py --runs-dir runs --as-of 2026-09-08

Exit codes: 0 GREEN, 1 RED, 2 no data in the window.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DAILY_RE = re.compile(r"^ops-(\d{4}-\d{2}-\d{2})\.jsonl$")
WEEKLY_GLOB = "ops-weekly-*.jsonl"

TERMINAL_STATUSES = {"complete", "failed", "budget_exhausted", "cancelled"}
NON_ORPHAN_STATUSES = TERMINAL_STATUSES | {"skipped", "infra-error"}
REQUIRED_DRILLS = ("redis-drill", "restart", "smoke")
P90_LIMIT_S = 300.0
DAILY_TOLERANCE = 2


def _parse_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read one JSONL file. Returns (rows, malformed_count). Never raises."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows, malformed
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
        else:
            malformed += 1
    return rows, malformed


def evaluate(runs_dir: Path, *, as_of: date, days: int = 7) -> tuple[list[str], str]:
    """Evaluate the window ending on as_of (inclusive). Returns (lines, status)."""
    window = [as_of - timedelta(days=offset) for offset in range(days)]
    window_set = {day.isoformat() for day in window}
    weekdays = sorted(day for day in window if day.weekday() < 5)

    daily_rows: list[dict[str, Any]] = []
    rows_by_date: dict[str, list[dict[str, Any]]] = {}
    weekly_rows: list[dict[str, Any]] = []
    malformed = 0
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("ops-*.jsonl")):
            match = DAILY_RE.match(path.name)
            if match is None:
                continue
            rows, bad = _parse_rows(path)
            malformed += bad
            for row in rows:
                stamp = str(row.get("date") or match.group(1))
                if stamp in window_set:
                    daily_rows.append(row)
                    rows_by_date.setdefault(stamp, []).append(row)
        for path in sorted(runs_dir.glob(WEEKLY_GLOB)):
            rows, bad = _parse_rows(path)
            malformed += bad
            for row in rows:
                if str(row.get("date", "")) in window_set:
                    weekly_rows.append(row)

    lines: list[str] = []
    if malformed:
        lines.append(f"[INFO] ignored {malformed} malformed JSONL line(s) in {runs_dir}")
    if not daily_rows and not weekly_rows:
        lines.append(
            f"no data: no daily or weekly rows in the {days}-day window "
            f"ending {as_of.isoformat()} (looked in {runs_dir})"
        )
        return lines, "NO-DATA"

    gates: list[bool] = []

    # 1. dailies -----------------------------------------------------------
    missing = [day.isoformat() for day in weekdays if day.isoformat() not in rows_by_date]
    eligible = [row for row in daily_rows if row.get("verdict") != "skipped"]
    skipped = len(daily_rows) - len(eligible)
    ok_count = sum(1 for row in eligible if row.get("verdict") == "as-expected")
    denom = len(eligible)
    need = max(0, denom - DAILY_TOLERANCE)
    dailies_pass = not missing and ok_count >= need
    gates.append(dailies_pass)
    tag = "PASS" if dailies_pass else "FAIL"
    lines.append(
        f"[{tag}] dailies: {ok_count}/{denom} as-expected "
        f"(need >={need}; skipped={skipped}) over weekdays "
        f"{', '.join(day.isoformat() for day in weekdays)}"
    )
    if missing:
        lines.append(f"       missing weekday file(s): {', '.join(missing)} (missing day = red)")

    # 2. orphans -----------------------------------------------------------
    orphan_daily = sum(
        1
        for row in daily_rows
        if row.get("timed_out") is True or str(row.get("status")) not in NON_ORPHAN_STATUSES
    )
    orphan_weekly = sum(
        int(row.get("orphans") or 0) for row in weekly_rows if row.get("check") == "restart"
    )
    orphans = orphan_daily + orphan_weekly
    orphans_pass = orphans == 0
    gates.append(orphans_pass)
    lines.append(
        f"[{'PASS' if orphans_pass else 'FAIL'}] orphans: {orphans} "
        f"non-terminal past deadline+60s (daily={orphan_daily} weekly-restart={orphan_weekly})"
    )

    # 3. p90 elapsed -------------------------------------------------------
    samples = sorted(
        float(row["elapsed_s"])
        for row in daily_rows
        if row.get("status") == "complete" and isinstance(row.get("elapsed_s"), (int, float))
    )
    if samples:
        rank = max(0, math.ceil(0.9 * len(samples)) - 1)
        p90 = samples[rank]
        p90_pass = p90 < P90_LIMIT_S
        gates.append(p90_pass)
        lines.append(
            f"[{'PASS' if p90_pass else 'FAIL'}] p90 final-report elapsed_s: "
            f"{p90:.1f}s over {len(samples)} COMPLETE rows (nearest-rank, limit <{P90_LIMIT_S:.0f}s)"
        )
    else:
        gates.append(False)
        lines.append("[FAIL] p90 final-report elapsed_s: no COMPLETE daily rows to measure")

    # 4. 500s --------------------------------------------------------------
    http_500 = sum(1 for row in [*daily_rows, *weekly_rows] if row.get("http_status") == 500)
    http_pass = http_500 == 0
    gates.append(http_pass)
    lines.append(f"[{'PASS' if http_pass else 'FAIL'}] http-500s recorded: {http_500}")

    # 5. drills ------------------------------------------------------------
    drills_pass = True
    for name in REQUIRED_DRILLS:
        named = [row for row in weekly_rows if row.get("check") == name]
        fails = sum(1 for row in named if row.get("verdict") == "fail")
        passes = sum(1 for row in named if row.get("verdict") == "pass")
        skips = sum(1 for row in named if row.get("verdict") == "skip")
        ok = fails == 0 and passes >= 1
        drills_pass = drills_pass and ok
        lines.append(
            f"[{'PASS' if ok else 'FAIL'}] drill {name}: "
            f"{passes} pass / {fails} fail / {skips} skip in window"
        )
    gates.append(drills_pass)

    status = "GREEN" if all(gates) else "RED"
    return lines, status


def reset_explanation(as_of: date, days: int) -> list[str]:
    return [
        f"reset: rolling {days}-day window ending {as_of.isoformat()}; "
        "a RED clears when fresh passing runs age the failure out of the window.",
        "reset: missing weekdays need a fresh --live daily; failed drills need a fresh --live weekly drill.",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Week-green checker for ARGUS ops routine.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--as-of",
        default=None,
        help="Window end date YYYY-MM-DD (default: today). For tests/backfill.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.as_of:
        try:
            as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
        except ValueError:
            print(f"Bad --as-of {args.as_of!r}: expected YYYY-MM-DD", file=sys.stderr)
            return 3
    else:
        as_of = datetime.now().date()
    if args.days < 7:
        print("--days must be >= 7 (a full ops week)", file=sys.stderr)
        return 3
    lines, status = evaluate(Path(args.runs_dir), as_of=as_of, days=args.days)
    for line in lines:
        print(line)
    for line in reset_explanation(as_of, args.days):
        print(line)
    print(f"overall: {status}")
    return {"GREEN": 0, "RED": 1, "NO-DATA": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
