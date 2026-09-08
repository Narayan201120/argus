"""P6-2 weekly ops drills (scripts + docs + tests only; no server changes).

Checks (select with --check, default all):
  redis-drill  Stop our argus-redis container, assert the server still
               answers GET /v1/models with 200 while /v1/health reports
               redis degraded, then start the container and assert health
               returns ok. Skips (never fails) when docker is missing, the
               argus-redis container is absent, or the server runs with
               redis disabled. The container is ours (named argus-redis);
               shared/external Redis instances are never touched.
  restart      SEMI-AUTO: starts one cheap investigation, prints "restart
               the ARGUS server now, then press Enter", waits, then
               verifies the row reached a terminal state and no orphan
               rows remain past deadline+60 s. Full auto-restart is
               intentionally NOT done: process supervision belongs to
               the operator.
  smoke        Shells out to scripts/smoke_live.py --live with a cheap
               subset (health models; +query only with --with-llm, since
               query spends LLM quota).

All rows are appended to runs/ops-weekly-YYYY-Www.jsonl.

Usage:
    python scripts/ops_weekly.py --dry-run
    python scripts/ops_weekly.py --live --check smoke
    ARGUS_OPS_LIVE=1 python scripts/ops_weekly.py --check redis-drill restart

Exit codes: 0 no failures (skips ok), 1 any check failed,
2 refused to run (guard), 3 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
REDIS_CONTAINER = "argus-redis"
RESTART_PROBE_QUERY = "ARGUS ops restart drill probe"
RESTART_ORPHAN_GRACE_S = 60.0

ALL_CHECKS = ("redis-drill", "restart", "smoke")
TERMINAL_STATUSES = {"complete", "failed", "budget_exhausted", "cancelled"}

SMOKE_SCRIPT = str(Path(__file__).resolve().parent / "smoke_live.py")


class OpsInfraError(RuntimeError):
    """Transport, HTTP-status, or protocol failure talking to the server."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def _today() -> str:
    return datetime.now().date().isoformat()


def _new_row(check: str, started: float) -> dict[str, Any]:
    return {
        "date": _today(),
        "check": check,
        "verdict": "fail",
        "detail": "",
        "elapsed_s": 0.0,
        "orphans": 0,
        "http_status": None,
        "_started": started,
    }


def _finish(row: dict[str, Any], verdict: str, detail: str, started: float) -> dict[str, Any]:
    row["verdict"] = verdict
    row["detail"] = detail
    row["elapsed_s"] = time.monotonic() - started
    row.pop("_started", None)
    return row


def _get_json(client: httpx.Client, path: str) -> Any:
    """GET path and return the JSON body. Raises OpsInfraError."""
    try:
        response = client.get(path, timeout=15.0)
    except httpx.HTTPError as exc:
        raise OpsInfraError(f"GET {path} failed: {exc}") from exc
    if response.status_code != 200:
        raise OpsInfraError(f"GET {path} HTTP {response.status_code}", response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise OpsInfraError(f"GET {path} returned non-JSON") from exc


def _poll_health_ok(
    client: httpx.Client, timeout_s: float, *, poll_s: float = 5.0
) -> tuple[bool, str]:
    """Poll /v1/health until status ok with redis ok. Returns (ok, detail)."""
    deadline = time.monotonic() + timeout_s
    last = "no poll yet"
    while True:
        try:
            payload = _get_json(client, "/v1/health")
        except OpsInfraError as exc:
            last = str(exc)
        else:
            if isinstance(payload, dict):
                status = payload.get("status")
                redis = payload.get("redis", "ok")
                if status == "ok" and redis in {"ok", "disabled"}:
                    return True, f"health ok (redis={redis})"
                last = f"status={status!r} redis={redis!r}"
            else:
                last = "health returned non-JSON"
        if time.monotonic() >= deadline:
            return False, f"health did not return ok within {timeout_s:.0f}s (last: {last})"
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))


# ── redis-drill ──────────────────────────────────────────────────────────────


def redis_drill(
    client: httpx.Client,
    *,
    run: Any = subprocess.run,
    with_llm: bool = False,
    recover_timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Stop argus-redis, assert degraded-but-answering, restart, assert ok."""
    started = time.monotonic()
    row = _new_row("redis-drill", started)
    try:
        version = run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return _finish(row, "skip", "docker CLI not present", started)
    except Exception as exc:  # noqa: BLE001 - drill must report, never traceback
        return _finish(row, "skip", f"docker probe failed: {exc}", started)
    if getattr(version, "returncode", 1) != 0:
        err = str(getattr(version, "stderr", "")).strip()[:200]
        return _finish(row, "skip", f"docker CLI not usable (rc={version.returncode}): {err}", started)

    try:
        health0 = _get_json(client, "/v1/health")
    except OpsInfraError as exc:
        return _finish(row, "fail", f"preflight GET /v1/health failed: {exc}", started)
    if isinstance(health0, dict) and health0.get("redis") == "disabled":
        return _finish(row, "skip", "server runs with redis disabled; drill N/A", started)
    try:
        models0 = client.get("/v1/models", timeout=15.0)
    except httpx.HTTPError as exc:
        return _finish(row, "fail", f"preflight GET /v1/models failed: {exc}", started)
    if models0.status_code != 200:
        return _finish(row, "fail", f"preflight GET /v1/models HTTP {models0.status_code}", started)

    stopped = run(["docker", "stop", REDIS_CONTAINER], capture_output=True, text=True, timeout=60)
    if getattr(stopped, "returncode", 1) != 0:
        err = str(getattr(stopped, "stderr", "")).strip()[:200]
        return _finish(
            row,
            "skip",
            f"docker stop {REDIS_CONTAINER} rc={stopped.returncode} ({err}); "
            "container absent? refusing to force shared infra",
            started,
        )

    failure: str | None = None
    try:
        try:
            models1 = client.get("/v1/models", timeout=15.0)
        except httpx.HTTPError as exc:
            failure = f"GET /v1/models during outage failed: {exc}"
        else:
            if models1.status_code != 200:
                failure = f"GET /v1/models during outage HTTP {models1.status_code}"
        if failure is None:
            try:
                health1 = _get_json(client, "/v1/health")
            except OpsInfraError as exc:
                failure = f"GET /v1/health during outage failed: {exc}"
                health1 = None
            if failure is None:
                if not isinstance(health1, dict):
                    failure = "GET /v1/health during outage returned non-JSON"
                elif health1.get("status") == "ok" and health1.get("redis", "ok") == "ok":
                    failure = f"health still ok with redis stopped: {str(health1)[:200]}"
        if failure is None and with_llm:
            try:
                qresp = client.post("/v1/query", json={"query": "redis drill probe"}, timeout=60.0)
            except httpx.HTTPError as exc:
                failure = f"POST /v1/query during outage failed: {exc}"
            else:
                if qresp.status_code != 200:
                    failure = f"POST /v1/query during outage HTTP {qresp.status_code}"
    finally:
        run(["docker", "start", REDIS_CONTAINER], capture_output=True, text=True, timeout=60)
        recovered, rec_detail = _poll_health_ok(client, recover_timeout_s)
        if not recovered:
            suffix = f"redis did not recover after start: {rec_detail}"
            failure = suffix if failure is None else f"{failure}; ALSO {suffix}"

    if failure is not None:
        return _finish(row, "fail", failure, started)
    extra = " +query" if with_llm else ""
    return _finish(
        row,
        "pass",
        f"models 200 during outage{extra}; health reported redis degraded; redis recovered ok",
        started,
    )


# ── restart drill (semi-auto) ────────────────────────────────────────────────


def restart_check(
    client: httpx.Client,
    *,
    prompt: Callable[[str], str] = input,
    limit: int = 50,
    wall_limit_s: float = 120.0,
    poll_s: float = 5.0,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Start an investigation, wait for a manual server restart, verify no orphans."""
    started = time.monotonic()
    row = _new_row("restart", started)
    try:
        created = client.post("/v1/investigate", json={"query": RESTART_PROBE_QUERY}, timeout=30.0)
    except httpx.HTTPError as exc:
        return _finish(row, "fail", f"investigate POST failed: {exc}", started)
    if created.status_code != 202:
        return _finish(row, "fail", f"investigate POST HTTP {created.status_code}", started)
    try:
        payload = created.json()
    except ValueError:
        return _finish(row, "fail", "investigate POST returned non-JSON", started)
    investigation_id = payload.get("investigation_id") if isinstance(payload, dict) else None
    if not investigation_id:
        return _finish(row, "fail", "investigate POST returned no investigation_id", started)

    print("Restart drill: an investigation is running.")
    print("restart the ARGUS server now, then press Enter")
    try:
        prompt("Press Enter after restarting the server (Ctrl-C aborts): ")
    except EOFError:
        return _finish(row, "fail", "prompt aborted (EOF); drill incomplete", started)

    poll_url = f"/v1/investigate/{investigation_id}"
    body: dict[str, Any] = {}
    terminal: str | None = None
    gone_after_restart = False
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            response = client.get(poll_url, timeout=15.0)
        except httpx.HTTPError as exc:
            return _finish(row, "fail", f"board poll failed after restart: {exc}", started)
        if response.status_code == 404:
            # In-memory server lost the row across the restart: no orphan,
            # nothing past deadline for this id. Still scan the list below.
            gone_after_restart = True
            break
        if response.status_code != 200:
            return _finish(row, "fail", f"board poll HTTP {response.status_code}", started)
        try:
            data = response.json()
        except ValueError:
            return _finish(row, "fail", "board poll returned non-JSON", started)
        body = data if isinstance(data, dict) else {}
        if body.get("status") in TERMINAL_STATUSES:
            terminal = str(body.get("status"))
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))

    try:
        listing = _get_json(client, f"/v1/investigations?limit={limit}")
    except OpsInfraError as exc:
        return _finish(row, "fail", f"list investigations failed after restart: {exc}", started)
    items = listing.get("investigations", []) if isinstance(listing, dict) else []
    now = time.time()
    orphans = 0
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("status") in TERMINAL_STATUSES:
            continue
        created_at = item.get("created_at")
        age = now - float(created_at) if isinstance(created_at, (int, float)) else 0.0
        if age > wall_limit_s + RESTART_ORPHAN_GRACE_S:
            orphans += 1
    row["orphans"] = orphans

    if gone_after_restart:
        detail = f"row {investigation_id} gone after restart (no orphan for this id); orphans={orphans}"
        if orphans:
            return _finish(row, "fail", detail, started)
        return _finish(row, "pass", detail, started)
    if terminal is None:
        return _finish(
            row,
            "fail",
            f"row {investigation_id} never reached terminal "
            f"(last status={body.get('status')!r}); orphans={orphans}",
            started,
        )
    detail = f"row {investigation_id} terminal={terminal}/{body.get('status_reason')}; orphans={orphans}"
    if orphans:
        return _finish(row, "fail", detail, started)
    return _finish(row, "pass", detail, started)


# ── smoke (shell-out) ────────────────────────────────────────────────────────


def smoke_check(
    *,
    runner: Any = subprocess.run,
    base_url: str = DEFAULT_BASE_URL,
    with_llm: bool = False,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Shell out to scripts/smoke_live.py --live with a cheap subset."""
    started = time.monotonic()
    row = _new_row("smoke", started)
    only = ["health", "models"] + (["query"] if with_llm else [])
    cmd = [sys.executable, SMOKE_SCRIPT, "--live", "--base-url", base_url, "--only", *only]
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - drill must report, never traceback
        return _finish(row, "fail", f"smoke subprocess failed: {exc}", started)
    tail = (str(getattr(proc, "stdout", "")) + str(getattr(proc, "stderr", "")))[-500:].strip()
    if getattr(proc, "returncode", 1) != 0:
        return _finish(row, "fail", f"smoke_live rc={proc.returncode}: {tail}", started)
    return _finish(row, "pass", f"smoke_live ok ({', '.join(only)}): {tail[-200:]}", started)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weekly ARGUS ops drills.")
    parser.add_argument("--base-url", default=os.environ.get("ARGUS_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--check",
        nargs="*",
        choices=ALL_CHECKS,
        default=list(ALL_CHECKS),
        help="Subset of drills to run (default: all).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run LLM-spending probes (query in smoke + redis drill).",
    )
    parser.add_argument("--wall-limit-s", type=float, default=120.0)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to actually run; alternatively set ARGUS_OPS_LIVE=1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned drills, run nothing (no docker, no network), exit 0.",
    )
    parser.add_argument(
        "--out", default=None, help="JSONL output path (default runs/ops-weekly-YYYY-Www.jsonl)."
    )
    return parser


def _default_out() -> str:
    today = datetime.now().date()
    year, week, _ = today.isocalendar()
    return f"runs/ops-weekly-{year}-W{week:02d}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = list(args.check)

    if args.dry_run:
        print(f"Dry run: {len(selected)} weekly drill(s) planned ({', '.join(selected)}).")
        if "redis-drill" in selected:
            print(f"  redis-drill: docker stop/start {REDIS_CONTAINER} + GET /v1/models + GET /v1/health")
        if "restart" in selected:
            print("  restart: POST /v1/investigate, print restart prompt, wait for Enter, verify terminal")
        if "smoke" in selected:
            subset = "health models" + (" query" if args.with_llm else "")
            print(f"  smoke: smoke_live.py --live --only {subset}")
            print("         (spends LLM quota only with --with-llm)")
        return 0

    unlocked = args.live or os.environ.get("ARGUS_OPS_LIVE", "").lower() in {"1", "true"}
    if not unlocked:
        print(
            "Refusing to run: weekly drills touch live infra (docker, server restart).\n"
            "Pass --live or set ARGUS_OPS_LIVE=1.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out) if args.out else Path(_default_out())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with httpx.Client(base_url=args.base_url, timeout=45.0) as client:
        for name in selected:
            try:
                if name == "redis-drill":
                    row = redis_drill(client, with_llm=args.with_llm)
                elif name == "restart":
                    row = restart_check(client, wall_limit_s=args.wall_limit_s)
                else:
                    row = smoke_check(base_url=args.base_url, with_llm=args.with_llm)
            except Exception as exc:  # noqa: BLE001 - drills report, never traceback
                row = {
                    "date": _today(),
                    "check": name,
                    "verdict": "fail",
                    "detail": f"drill raised: {exc}",
                    "elapsed_s": 0.0,
                    "orphans": 0,
                    "http_status": None,
                }
            rows.append(row)
            print(f"[{row['verdict'].upper()}] {name} {row['detail']}")

    with out_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    failed = sum(1 for row in rows if row["verdict"] == "fail")
    print(f"Ops weekly: {len(rows) - failed}/{len(rows)} drills passed (rows in {out_path}).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
