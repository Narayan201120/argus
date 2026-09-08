# ARGUS ops routine (P6-2)

Daily probe + weekly drills + week-green checker. Scripts only — no server
code. All state lands in `runs/` (operator-local JSONL, not committed).

> Note: `runs/` is currently **not** in `.gitignore`. Per the task brief the
> scripts write there anyway and nothing was added to `.gitignore`. Do not
> commit `runs/` contents.

## Daily — `scripts/ops_daily.py`

Fixed 5-query set, one JSONL row per query in `runs/ops-YYYY-MM-DD.jsonl`:

| kind | query | expectation |
|---|---|---|
| radar | dense vs sparse retrieval scientific QA | COMPLETE (or documented-budget) |
| rag | ARGUS response-cache TTL policy | COMPLETE (or documented-budget) |
| web | latest Tavily API pricing | COMPLETE (or documented-budget); **skipped** when web tools are off |
| poisoned | Prove P=NP using our library | FAILED / provider_failure within 60 s |
| dupe | large language models industrial applications | COMPLETE (budget states do not count) |

Documented-budget = `budget_exhausted` with reason `cost_limit`,
`iteration_limit`, `tool_call_limit`, or `wall_clock_limit`.

```powershell
venv\Scripts\python.exe scripts\ops_daily.py --dry-run
venv\Scripts\python.exe scripts\ops_daily.py --live
$env:ARGUS_OPS_LIVE=1; venv\Scripts\python.exe scripts\ops_daily.py --only radar poisoned
$env:ARGUS_OPS_SKIP_WEB=1; venv\Scripts\python.exe scripts\ops_daily.py --live  # WEB_TOOLS off host
```

Flags: `--live` / `ARGUS_OPS_LIVE=1` guard (exit 2 without it),
`--dry-run` (planned POSTs + spend ceiling, no network), `--only` kinds,
`--base-url` (default `http://127.0.0.1:8001`), `--poll-s`, `--timeout-s`
(180 s per query), `--spend-cap-usd` (default 2.00, exit 2 over cap),
`--out`.

Exits: 0 all-as-expected (skips ok) · 1 any mismatch/infra row ·
2 guard/cap refusal · 3 usage error.

## Weekly — `scripts/ops_weekly.py`

```powershell
venv\Scripts\python.exe scripts\ops_weekly.py --dry-run
venv\Scripts\python.exe scripts\ops_weekly.py --live --check smoke
venv\Scripts\python.exe scripts\ops_weekly.py --live                          # all three drills
venv\Scripts\python.exe scripts\ops_weekly.py --live --check smoke --with-llm # spend LLM quota
```

- `redis-drill`: `docker stop argus-redis` → models still 200, health
  reports redis degraded → `docker start argus-redis` → health ok.
  Skips when docker is missing, the container is absent, or the server
  runs with redis disabled. Only the `argus-redis` container is touched.
- `restart` (SEMI-AUTO): posts an investigation, prints
  `restart the ARGUS server now, then press Enter`, waits, then verifies
  the row is terminal and no orphans remain past deadline+60 s.
  Full auto-restart is intentionally not scripted — process supervision
  belongs to the operator.
- `smoke`: shells out to `scripts/smoke_live.py --live --only health
  models` (`+ query` with `--with-llm`).

Rows append to `runs/ops-weekly-YYYY-Www.jsonl`. Exits: 0 no failures
(skips ok) · 1 any drill failed · 2 guard refusal · 3 usage error.

## Checker — `scripts/green_week.py`

```powershell
venv\Scripts\python.exe scripts\green_week.py
venv\Scripts\python.exe scripts\green_week.py --as-of 2026-09-08
```

Five gates over the rolling 7-day window: ≥23/25 weekday dailies
as-expected (missing weekday = RED; skipped rows leave the denominator
and the bar becomes denominator−2) · zero non-terminal past
deadline+60 s · p90 COMPLETE `elapsed_s` < 300 s (nearest-rank) ·
zero `http_status` 500 rows · each drill ≥1 pass and zero fails.
Prints one line per gate, the overall GREEN/RED, and how a RED resets
(rolling window: fresh passing runs age the failure out).

Exits: 0 GREEN · 1 RED · 2 no data.

## Unattended runs (Windows Task Scheduler)

Daily probe + weekly cheap drills run unattended. The `restart` drill
stays manual (it waits for Enter). Run the mechanical parts on a
schedule, do `restart` by hand once a week, then judge green.

```powershell
schtasks /Create /TN "ARGUS ops daily" /TR "A:\Projects\argus\venv\Scripts\python.exe A:\Projects\argus\scripts\ops_daily.py --live" /SC DAILY /ST 08:00
schtasks /Create /TN "ARGUS ops weekly drills" /TR "A:\Projects\argus\venv\Scripts\python.exe A:\Projects\argus\scripts\ops_weekly.py --live --check redis-drill smoke" /SC WEEKLY /D MON /ST 09:00
schtasks /Create /TN "ARGUS week green" /TR "A:\Projects\argus\venv\Scripts\python.exe A:\Projects\argus\scripts\green_week.py" /SC WEEKLY /D FRI /ST 17:00
```

Set `ARGUS_OPS_LIVE=1` via `/RU` task credentials environment or keep
`--live` on the command line as above (the flag is already in the
one-liners). `--base-url` overrides the default when the server is not
on `127.0.0.1:8001`.

## What stays manual

- Provider invoices / quota top-ups (scripts only print a client-side
  spend *ceiling*; real billing lives in provider consoles).
- Corpus uploads and eval schedule changes.
- RAG / Radar upgrades and config changes (`WEB_TOOLS`, keys, URLs).
- The `restart` drill itself (semi-auto by design).
- Judging week-green on process failures the scripts cannot see
  (missed manual steps, ignored RED days, unrecorded incidents).

## Exit criteria recap (from plan)

Week is GREEN when `green_week.py` exits 0: ≥23/25 weekday dailies
as-expected with no missing weekday, zero orphans past deadline+60 s,
p90 final-report under 300 s, zero 500s, all three drills passing in
the window. Any RED must name its gate and the reset path (re-run the
failing daily/drill; the rolling window clears it within 7 days).
