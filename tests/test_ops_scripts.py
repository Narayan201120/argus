"""P6-2 ops scripts tests (mock-only; never touches network or docker).

httpx.Client / subprocess.run / input are replaced with fakes, so any
attempted real I/O fails loudly.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts import green_week, ops_daily, ops_weekly


class _Resp:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)[:200]

    def json(self) -> Any:
        return self._payload


class _ExplodingClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("network attempted in a no-network test")


def _locked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARGUS_OPS_LIVE", raising=False)
    monkeypatch.delenv("ARGUS_OPS_SKIP_WEB", raising=False)


# ── ops_daily fakes ──────────────────────────────────────────────────────────


def _board(status: str, reason: str | None, ev: int = 2, cl: int = 1) -> dict[str, Any]:
    return {
        "status": status,
        "status_reason": reason,
        "evidence": [{"id": f"ev-{i}"} for i in range(ev)],
        "claims": [
            {
                "id": f"cl-{i}",
                "statement": "canned",
                "confidence": 0.8,
                "evidence_ids": ["ev-0"] if ev else [],
                "status": "supported",
            }
            for i in range(cl)
        ],
        "created_at": 1700000000.0,
        "updated_at": 1700000010.0,
    }


def _good_boards() -> dict[str, dict[str, Any]]:
    return {
        "radar": _board("complete", "sufficient_evidence"),
        "rag": _board("complete", "sufficient_evidence"),
        "web": _board("complete", "sufficient_evidence"),
        "poisoned": _board("failed", "provider_failure", ev=0, cl=0),
        "dupe": _board("complete", "sufficient_evidence"),
    }


class _DailyClient:
    """Fake httpx.Client: polls return the scripted board for the last POSTed query."""

    def __init__(
        self,
        boards: dict[str, dict[str, Any]],
        *args: Any,
        health: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._boards = boards
        self._health = health or {"status": "ok", "connectors": [{"connector_id": "x"}], "redis": "ok"}
        self._meta = meta if meta is not None else {"workspace": {"web": True}}
        self._last_query = ""
        self.posts: list[str] = []

    def __enter__(self) -> _DailyClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, path: str, **kwargs: Any) -> _Resp:
        payload = kwargs.get("json", {})
        self._last_query = str(payload.get("query", ""))
        self.posts.append(self._last_query)
        return _Resp(202, {"investigation_id": "inv-1"})

    def get(self, path: str, **kwargs: Any) -> _Resp:
        if path == "/v1/health":
            return _Resp(200, self._health)
        if path == "/v1/meta":
            return _Resp(200, self._meta)
        for item in ops_daily.DAILY_QUERIES:
            if item["query"] == self._last_query:
                return _Resp(200, self._boards[item["kind"]])
        return _Resp(200, self._boards["radar"])


def _run_daily(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    boards: dict[str, dict[str, Any]],
    argv: list[str],
    *,
    health: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    monkeypatch.setenv("ARGUS_OPS_LIVE", "1")
    monkeypatch.delenv("ARGUS_OPS_SKIP_WEB", raising=False)
    out = tmp_path / "ops.jsonl"
    monkeypatch.setattr(
        ops_daily.httpx,
        "Client",
        lambda *a, **k: _DailyClient(boards, *a, health=health, meta=meta, **k),
    )
    monkeypatch.setattr(ops_daily.time, "sleep", lambda s: None)
    rc = ops_daily.main([*argv, "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rc, rows


# ── ops_daily: guard / dry-run ───────────────────────────────────────────────


def test_daily_guard_exit_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _locked(monkeypatch)
    monkeypatch.setattr(ops_daily.httpx, "Client", _ExplodingClient)
    assert ops_daily.main([]) == 2
    assert "Refusing to run" in capsys.readouterr().err


def test_daily_dry_run_no_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    monkeypatch.setattr(ops_daily.httpx, "Client", _ExplodingClient)
    assert ops_daily.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "5 investigate POST(s) planned" in out
    assert "ceiling estimate" in out


def test_daily_dry_run_only_subset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    monkeypatch.setattr(ops_daily.httpx, "Client", _ExplodingClient)
    assert ops_daily.main(["--dry-run", "--only", "radar"]) == 0
    assert "1 investigate POST(s) planned" in capsys.readouterr().out


def test_daily_unknown_kind_is_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)
    monkeypatch.setattr(ops_daily.httpx, "Client", _ExplodingClient)
    assert ops_daily.main(["--dry-run", "--only", "nope"]) == 3
    assert "Unknown kinds" in capsys.readouterr().err


# ── ops_daily: expectation matching ──────────────────────────────────────────


def test_daily_all_as_expected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, rows = _run_daily(monkeypatch, tmp_path, _good_boards(), ["--live"])
    assert rc == 0
    assert [row["kind"] for row in rows] == ["radar", "rag", "web", "poisoned", "dupe"]
    assert all(row["verdict"] == "as-expected" for row in rows)
    for row in rows:
        for key in ("date", "query", "kind", "status", "status_reason", "evidence", "claims",
                    "elapsed_s", "cost_note"):
            assert key in row, (row["kind"], key)
    out = capsys.readouterr().out
    assert "5/5 as-expected-or-skipped" in out


def test_daily_poisoned_complete_is_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    boards = _good_boards()
    boards["poisoned"] = _board("complete", "sufficient_evidence")
    rc, rows = _run_daily(monkeypatch, tmp_path, boards, ["--live"])
    assert rc == 1
    poisoned = next(row for row in rows if row["kind"] == "poisoned")
    assert poisoned["verdict"] == "mismatch"
    assert "expected FAILED" in poisoned["note"]


def test_daily_expectation_units() -> None:
    verdict, _ = ops_daily.check_expectations(
        "poisoned", status="failed", status_reason="provider_failure", elapsed_s=3.0
    )
    assert verdict == "as-expected"
    verdict, note = ops_daily.check_expectations(
        "poisoned", status="failed", status_reason="provider_failure", elapsed_s=61.0
    )
    assert verdict == "mismatch" and "60s" in note
    verdict, _ = ops_daily.check_expectations(
        "poisoned", status="failed", status_reason="wrong_reason", elapsed_s=3.0
    )
    assert verdict == "mismatch"
    verdict, _ = ops_daily.check_expectations(
        "radar", status="budget_exhausted", status_reason="cost_limit", elapsed_s=9.0
    )
    assert verdict == "as-expected"
    verdict, _ = ops_daily.check_expectations(
        "dupe", status="budget_exhausted", status_reason="cost_limit", elapsed_s=9.0
    )
    assert verdict == "mismatch"


def test_daily_web_skip_via_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARGUS_OPS_LIVE", "1")
    monkeypatch.setenv("ARGUS_OPS_SKIP_WEB", "1")
    out = tmp_path / "ops.jsonl"
    monkeypatch.setattr(ops_daily.httpx, "Client", lambda *a, **k: _DailyClient(_good_boards()))
    monkeypatch.setattr(ops_daily.time, "sleep", lambda s: None)
    assert ops_daily.main(["--live", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    web = next(row for row in rows if row["kind"] == "web")
    assert web["verdict"] == "skipped" and web["status"] == "skipped"


def test_daily_web_skip_via_meta_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    meta = {"workspace": {"web": False}}
    rc, rows = _run_daily(monkeypatch, tmp_path, _good_boards(), ["--live"], meta=meta)
    assert rc == 0
    web = next(row for row in rows if row["kind"] == "web")
    assert web["verdict"] == "skipped"


# ── ops_weekly fakes ─────────────────────────────────────────────────────────


class _Proc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _RedisClient:
    def __init__(self, healths: list[dict[str, Any]]) -> None:
        self._healths = list(healths)

    def get(self, path: str, **kwargs: Any) -> _Resp:
        if path == "/v1/health":
            payload = self._healths[0] if len(self._healths) == 1 else self._healths.pop(0)
            return _Resp(200, payload)
        return _Resp(200, {"connectors": [], "total": 2})

    def post(self, path: str, **kwargs: Any) -> _Resp:
        return _Resp(
            200,
            {"result": "ok", "short_circuited": True, "model_statuses": [],
             "role_assignments": {"direct": "x"}},
        )


def _ok_health(redis: str = "ok", status: str = "ok") -> dict[str, Any]:
    return {"status": status, "connectors": [{"connector_id": "x"}], "redis": redis}


class _Docker:
    def __init__(self, stop_rc: int = 0) -> None:
        self.cmds: list[list[str]] = []
        self.stop_rc = stop_rc

    def __call__(self, cmd: list[str], **kwargs: Any) -> _Proc:
        self.cmds.append(cmd)
        if cmd[:2] == ["docker", "version"]:
            return _Proc(0, stdout="27.0.0")
        if cmd[:2] == ["docker", "stop"]:
            return _Proc(self.stop_rc, stderr="" if self.stop_rc == 0 else "No such container")
        if cmd[:2] == ["docker", "start"]:
            return _Proc(0)
        raise AssertionError(f"unexpected docker cmd {cmd}")


class _RestartClient:
    def __init__(self, board: Any, listing: dict[str, Any]) -> None:
        self._board = board
        self._listing = listing

    def post(self, path: str, **kwargs: Any) -> _Resp:
        return _Resp(202, {"investigation_id": "inv-9"})

    def get(self, path: str, **kwargs: Any) -> _Resp:
        if path.startswith("/v1/investigations"):
            return _Resp(200, self._listing)
        if isinstance(self._board, list):
            return self._board.pop(0)
        return self._board


# ── ops_weekly: guard / dry-run / drills ─────────────────────────────────────


def test_weekly_guard_exit_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _locked(monkeypatch)
    assert ops_weekly.main([]) == 2
    assert "Refusing to run" in capsys.readouterr().err


def test_weekly_dry_run_no_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _locked(monkeypatch)

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("side effect attempted in dry-run")

    monkeypatch.setattr(ops_weekly.subprocess, "run", _explode)
    monkeypatch.setattr(ops_weekly.httpx, "Client", _ExplodingClient)
    assert ops_weekly.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "redis-drill" in out and "restart" in out and "smoke" in out


def test_redis_drill_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops_weekly.time, "sleep", lambda s: None)
    client = _RedisClient([_ok_health(), _ok_health(redis="unavailable", status="degraded"), _ok_health()])
    docker = _Docker()
    row = ops_weekly.redis_drill(client, run=docker)  # type: ignore[arg-type]
    assert row["verdict"] == "pass", row
    stops = [cmd for cmd in docker.cmds if cmd[:2] == ["docker", "stop"]]
    starts = [cmd for cmd in docker.cmds if cmd[:2] == ["docker", "start"]]
    assert stops == [["docker", "stop", "argus-redis"]]
    assert starts == [["docker", "start", "argus-redis"]]


def test_redis_drill_docker_missing() -> None:
    def _missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("no docker")

    row = ops_weekly.redis_drill(_RedisClient([_ok_health()]), run=_missing)  # type: ignore[arg-type]
    assert row["verdict"] == "skip"
    assert "docker CLI not present" in row["detail"]


def test_redis_drill_redis_disabled_skips_without_stop() -> None:
    docker = _Docker()
    row = ops_weekly.redis_drill(_RedisClient([_ok_health(redis="disabled")]), run=docker)  # type: ignore[arg-type]
    assert row["verdict"] == "skip"
    assert "redis disabled" in row["detail"]
    assert not [cmd for cmd in docker.cmds if cmd[:2] == ["docker", "stop"]]


def test_restart_prompt_path_pass(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(ops_weekly.time, "sleep", lambda s: None)
    board = _Resp(200, {**_board("complete", "sufficient_evidence"), "investigation_id": "inv-9"})
    client = _RestartClient(board, {"investigations": []})
    row = ops_weekly.restart_check(client, prompt=lambda _: "")  # type: ignore[arg-type]
    assert row["verdict"] == "pass", row
    assert row["orphans"] == 0
    assert "restart the ARGUS server now" in capsys.readouterr().out


def test_restart_gone_after_restart_is_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops_weekly.time, "sleep", lambda s: None)
    client = _RestartClient(_Resp(404, "gone"), {"investigations": []})
    row = ops_weekly.restart_check(client, prompt=lambda _: "")  # type: ignore[arg-type]
    assert row["verdict"] == "pass", row
    assert "gone after restart" in row["detail"]


def test_smoke_shellout_pass_and_fail() -> None:
    ok = ops_weekly.smoke_check(runner=lambda *a, **k: _Proc(0, stdout="Smoke OK"), base_url="http://x")
    assert ok["verdict"] == "pass" and ok["check"] == "smoke"
    bad = ops_weekly.smoke_check(
        runner=lambda *a, **k: _Proc(1, stdout="", stderr="boom"), base_url="http://x"
    )
    assert bad["verdict"] == "fail"
    assert "rc=1" in bad["detail"]


# ── green_week fixtures ──────────────────────────────────────────────────────


AS_OF = date(2026, 9, 8)


def _window_weekdays() -> list[str]:
    days = [(AS_OF - timedelta(days=i)).isoformat() for i in range(7)]
    return sorted(day for day in days if date.fromisoformat(day).weekday() < 5)


def _daily_row(day: str, kind: str, verdict: str = "as-expected") -> dict[str, Any]:
    poisoned = kind == "poisoned"
    return {
        "date": day,
        "query": kind,
        "kind": kind,
        "investigation_id": "inv-1",
        "status": "failed" if poisoned else "complete",
        "status_reason": "provider_failure" if poisoned else "sufficient_evidence",
        "evidence": 0 if poisoned else 2,
        "claims": 0 if poisoned else 1,
        "elapsed_s": 3.0 if poisoned else 12.0,
        "timed_out": False,
        "verdict": verdict,
        "note": "",
        "cost_note": "",
        "created_at": 1700000000.0,
        "updated_at": 1700000010.0,
        "http_status": None,
    }


def _write_green_fixture(tmp_path: Path) -> None:
    day = _window_weekdays()[0]
    for stamp in _window_weekdays():
        rows = [_daily_row(stamp, kind) for kind in ("radar", "rag", "web", "poisoned", "dupe")]
        (tmp_path / f"ops-{stamp}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    weekly = [
        {"date": day, "check": name, "verdict": "pass", "detail": "ok",
         "elapsed_s": 1.0, "orphans": 0, "http_status": None}
        for name in ("redis-drill", "restart", "smoke")
    ]
    (tmp_path / "ops-weekly-2026-W37.jsonl").write_text(
        "\n".join(json.dumps(row) for row in weekly) + "\n", encoding="utf-8"
    )


def test_green_week_green(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _window_weekdays() == ["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"]
    _write_green_fixture(tmp_path)
    rc = green_week.main(["--runs-dir", str(tmp_path), "--as-of", "2026-09-08"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "overall: GREEN" in out
    assert "25/25 as-expected" in out
    assert "reset:" in out


def test_green_week_red_on_mismatches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_green_fixture(tmp_path)
    for stamp in _window_weekdays()[:3]:
        path = tmp_path / f"ops-{stamp}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["kind"] == "radar":
                row["verdict"] = "mismatch"
                row["status"] = "failed"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    rc = green_week.main(["--runs-dir", str(tmp_path), "--as-of", "2026-09-08"])
    assert rc == 1
    assert "overall: RED" in capsys.readouterr().out


def test_green_week_red_on_missing_day(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_green_fixture(tmp_path)
    (tmp_path / f"ops-{_window_weekdays()[0]}.jsonl").unlink()
    rc = green_week.main(["--runs-dir", str(tmp_path), "--as-of", "2026-09-08"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "missing weekday" in out
    assert "overall: RED" in out


def test_green_week_red_on_500(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_green_fixture(tmp_path)
    path = tmp_path / f"ops-{_window_weekdays()[0]}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["http_status"] = 500
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    rc = green_week.main(["--runs-dir", str(tmp_path), "--as-of", "2026-09-08"])
    assert rc == 1
    assert "http-500s recorded: 1" in capsys.readouterr().out


def test_green_week_no_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = green_week.main(["--runs-dir", str(tmp_path), "--as-of", "2026-09-08"])
    assert rc == 2
    assert "no data" in capsys.readouterr().out
