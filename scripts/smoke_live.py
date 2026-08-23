"""Live smoke test for a running ARGUS instance.

Exercises /v1/health, /v1/models, /v1/query and - optionally - the SSE
stream and deep-report endpoints against a deployed server whose provider
keys are real. The script itself never needs API keys.

Per DEC-008 this script refuses to run unless explicitly unlocked with
--live or ARGUS_SMOKE_LIVE=1, because live runs spend provider tokens.

Usage:
    python scripts/smoke_live.py --base-url http://127.0.0.1:8000 --live
    ARGUS_SMOKE_LIVE=1 python scripts/smoke_live.py --only health models
    ARGUS_SMOKE_TOKEN=<jwt> python scripts/smoke_live.py --live

Exit codes: 0 all checks passed, 1 failures detected,
2 refused to run (guard), 3 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUERY = "What is the ARGUS architecture?"
REPORT_TIMEOUT_S = 300

ALL_CHECKS = ("health", "models", "query", "stream", "report", "audio")


# ── Pure validators (unit-testable without network) ─────────────────────────


def validate_health(payload: dict) -> list[str]:
    failures: list[str] = []
    if payload.get("status") not in {"ok", "degraded"}:
        failures.append(f"health.status={payload.get('status')!r} not in {{ok, degraded}}")
    connectors = payload.get("connectors")
    if not isinstance(connectors, list) or not connectors:
        failures.append("health.connectors is empty or missing")
    return failures


def validate_models(payload: dict) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload.get("connectors"), list):
        failures.append("models.connectors is missing")
    total = payload.get("total", 0)
    if not isinstance(total, int) or total <= 0:
        failures.append(f"models.total={total!r} is not a positive int")
    return failures


def validate_query_response(payload: dict) -> list[str]:
    failures: list[str] = []
    if not payload.get("result"):
        failures.append("query.result is empty")
    if payload.get("short_circuited") is False and not payload.get("model_statuses"):
        failures.append("parallel query returned no model_statuses")
    assignments = payload.get("role_assignments")
    if not isinstance(assignments, dict) or not assignments:
        failures.append("query.role_assignments is empty or missing")
    return failures


def validate_final_event(data: dict) -> list[str]:
    failures = validate_query_response(data)
    if data.get("router_strategy") not in {"static", "semantic"}:
        failures.append(f"final.router_strategy={data.get('router_strategy')!r} unexpected")
    return failures


def validate_transcription(payload: dict) -> list[str]:
    failures: list[str] = []
    if not payload.get("text"):
        failures.append("transcription.text is empty")
    if not payload.get("model"):
        failures.append("transcription.model is missing")
    return failures


VALIDATORS = {
    "health": validate_health,
    "models": validate_models,
}


# ── Network wrappers ────────────────────────────────────────────────────────


def run_simple_check(client: httpx.Client, name: str, path: str, method: str = "GET",
                     json_body: dict | None = None) -> list[str]:
    try:
        response = client.request(method, path, json=json_body)
    except httpx.HTTPError as exc:
        return [f"{name}: request failed: {exc}"]
    if response.status_code != 200:
        return [f"{name}: HTTP {response.status_code}: {response.text[:200]}"]
    return VALIDATORS[name](response.json())


def run_stream_check(client: httpx.Client, query: str) -> list[str]:
    events: dict[str, dict] = {}
    try:
        with client.stream(
            "POST", "/v1/query/stream", json={"query": query}, timeout=120.0
        ) as response:
            if response.status_code != 200:
                return [f"stream: HTTP {response.status_code}"]
            event_name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_name:
                    raw = line.split(":", 1)[1].strip()
                    try:
                        events[event_name] = json.loads(raw)
                    except json.JSONDecodeError:
                        events[event_name] = {}
    except httpx.HTTPError as exc:
        return [f"stream: request failed: {exc}"]

    failures: list[str] = []
    if "role_complete" not in events:
        failures.append("stream: no role_complete event observed")
    if "final" not in events:
        failures.append("stream: terminal final event missing")
        return failures
    failures.extend(f"stream.final.{f}" for f in validate_final_event(events["final"]))
    return failures


def run_report_check(client: httpx.Client, query: str) -> list[str]:
    try:
        created = client.post("/v1/report", json={"query": query}, timeout=30.0)
        if created.status_code != 202:
            return [f"report: HTTP {created.status_code}: {created.text[:200]}"]
        job_id = created.json().get("job_id")
        if not job_id:
            return ["report: no job_id in 202 response"]

        deadline = time.monotonic() + REPORT_TIMEOUT_S
        poll_url = f"/v1/report/{job_id}"
        while time.monotonic() < deadline:
            status = client.get(poll_url, timeout=15.0)
            if status.status_code != 200:
                return [f"report: poll HTTP {status.status_code}"]
            body = status.json()
            if body.get("status") == "failed":
                return [f"report: job failed: {body.get('error')!r}"]
            if body.get("status") == "completed":
                if not body.get("result_markdown"):
                    return ["report: completed with empty result_markdown"]
                return []
            time.sleep(2.0)
        return [f"report: timed out after {REPORT_TIMEOUT_S}s"]
    except httpx.HTTPError as exc:
        return [f"report: request failed: {exc}"]


def run_audio_check(client: httpx.Client) -> list[str]:
    """Upload a small audio file to /v1/transcribe.

    Requires ARGUS_SMOKE_AUDIO_FILE to point at a short (<30s) wav/mp3.
    Skips (with a note, no failure) when the variable is unset - credits
    are never spent implicitly.
    """
    audio_path = os.environ.get("ARGUS_SMOKE_AUDIO_FILE")
    if not audio_path:
        print("(audio check skipped: set ARGUS_SMOKE_AUDIO_FILE to a <30s file)")
        return []

    try:
        with open(audio_path, "rb") as handle:
            content = handle.read()
    except OSError as exc:
        return [f"audio: cannot read ARGUS_SMOKE_AUDIO_FILE: {exc}"]

    filename = os.path.basename(audio_path)
    try:
        response = client.post("/v1/transcribe", files={"file": (filename, content)})
    except httpx.HTTPError as exc:
        return [f"audio: request failed: {exc}"]
    if response.status_code != 200:
        return [f"audio: HTTP {response.status_code}: {response.text[:200]}"]
    return [f"audio.transcription.{f}" for f in validate_transcription(response.json())]


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live smoke test for ARGUS.")
    parser.add_argument("--base-url", default=os.environ.get("ARGUS_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--token", default=os.environ.get("ARGUS_SMOKE_TOKEN"))
    parser.add_argument(
        "--only", nargs="*", choices=ALL_CHECKS, default=list(ALL_CHECKS),
        help="Subset of checks to run.",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Required to actually run; alternatively set ARGUS_SMOKE_LIVE=1.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    unlocked = args.live or os.environ.get("ARGUS_SMOKE_LIVE", "").lower() in {"1", "true"}
    if not unlocked:
        print(
            "Refusing to run: live smoke spends real provider tokens.\n"
            "Pass --live or set ARGUS_SMOKE_LIVE=1 once the server has keys.",
            file=sys.stderr,
        )
        return 2

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    selected = set(args.only)
    failures: list[str] = []

    with httpx.Client(base_url=args.base_url, headers=headers, timeout=45.0) as client:
        if "health" in selected:
            failures += run_simple_check(client, "health", "/v1/health")
        if "models" in selected:
            failures += run_simple_check(client, "models", "/v1/models")
        if "query" in selected:
            failures += run_simple_check(client, "query", "/v1/query", method="POST",
                                         json_body={"query": args.query})
        if "stream" in selected:
            failures += run_stream_check(client, args.query)
        if "report" in selected:
            failures += run_report_check(client, args.query)
        if "audio" in selected:
            failures += run_audio_check(client)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"Smoke FAILED with {len(failures)} problem(s).", file=sys.stderr)
        return 1

    ran = ", ".join(sorted(selected))
    print(f"Smoke OK ({ran}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
