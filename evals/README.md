# Evals

Human-graded evals for ARGUS answers. No LLM judge anywhere in this pipeline.
A person reads each run and assigns the verdict. Start here before running
anything that spends money.

## Corpus format

`corpus.yaml` is a YAML list. One item per case:

```yaml
- id: radar-01
  bucket: radar-answerable
  query: "How does dense passage retrieval score query-document pairs?"
  tools: [radar_search]
  expect: {min_evidence: 2, verdict: supported, must_cite: true}
  notes: "Why this case exists."
```

`expect.verdict` records the honest answer shape for the case. It never
constrains the pipeline. Any terminal state can pass when the board shows
zero fabrication.

## Buckets

20 cases, 5 per bucket.

- `radar-answerable`. CS-paper index staples. Retrieval scoring, RAG metrics,
  semantic routing, hybrid search, chunk overlap. These have stable answers.
- `rag-answerable`. Questions about the ARGUS repo itself. Architecture,
  cache TTL, rate limiting, board truncation caps, connector pins.
- `needs-web`. Facts current as of Sep 2026. Vendor pricing, fast-moving
  docs, release notes. Mock runs answer from canned evidence. Only a live
  run with web access can fully ground these.
- `unanswerable`. Future facts, private data, open problems, secrets,
  unrecorded events. The only honest output is refusal with zero claims.

## Workflows

Three scripts, three cost profiles. Pick the cheapest one that answers your
question.

Mock first, always. It spends nothing by construction. No HTTP client, no
provider SDK, canned boards built in-process:

```sh
venv/Scripts/python.exe scripts/eval_mock.py
venv/Scripts/python.exe scripts/eval_mock.py --only radar-01 rag-01
```

Live only when you mean it. `eval_live.py` POSTs to a running server, polls
each investigation to a terminal state, and appends one JSONL row per case
to `runs/<ts>.jsonl`. It refuses to run unless unlocked, because live runs
spend provider tokens:

```sh
venv/Scripts/python.exe scripts/eval_live.py --dry-run
venv/Scripts/python.exe scripts/eval_live.py --live --max-runs 3
```

Dry run prints the planned POST count and the pessimistic spend ceiling,
then exits without touching the server. The ceiling is per-run estimate
times planned runs, and the run aborts when the ceiling tops
`--spend-cap-usd` (default $2.00).

Grade last, by hand. `eval_grade.py` walks the ungraded rows of a run,
shows query plus synthesis text plus board counts, and prompts for a
verdict on stdin:

```sh
venv/Scripts/python.exe scripts/eval_grade.py grade --run runs/20260908T120000Z.jsonl
venv/Scripts/python.exe scripts/eval_grade.py grade --run runs/x.jsonl --only radar-01
```

Judgments append to `runs/<ts>.grades.jsonl`, one object per line with
`case_id`, `investigation_id`, `verdict`, `notes`, `graded_at`. Reruns skip
rows that already have a grade, so an interrupted session resumes where it
stopped. The script prints the verdict histogram at the end. For scripted
use there is a non-interactive mode:

```sh
venv/Scripts/python.exe scripts/eval_grade.py grade --run runs/x.jsonl \
  --non-interactive --verdict supported
```

Diff runs in CI. `eval_diff.py` compares a candidate run against
`evals/baselines/v0.6.0.json` on aggregates only. It never string-matches
report text. Three gates: no verdict drops, grounding drop within 5pp, p90
final-report latency within +25%.

## Rubric

Read the row, then grade what the system actually produced, not what the
corpus expected. Four verdicts:

- `supported`. Every claim is grounded. Each claim cites at least one
  evidence id, every cited id sits on the board, and no source is invented.
  This is the only passing verdict for answerable cases with claims.
- `partial`. At least half the claims are grounded. Use it for answers that
  mix solid cited work with ungrounded filler, and for needs-web cases where
  canned evidence only goes so far.
- `unsupported`. Invented sources, or mostly ungrounded claims. A confident
  answer with dangling citations lands here.
- `refused-correctly`. The run produced no fabrication and said so. Terminal
  state can be anything, complete included. The board is empty, or carries
  only `proposed` claims that cite nothing, and the synthesis states the
  limits instead of answering. Any invented reference vetoes this verdict.

Claim-level rule of thumb: a claim needs at least one cited evidence id
present on the board to count as grounded. Partial means half or more of
the claims clear that bar.

## Verdict semantics (DEC-054)

Verdicts never constrain terminal state. A `budget_exhausted` run with an
empty board and an honest synthesis is `refused-correctly` and passes. A
`complete` run with fabricated citations is `unsupported` and fails. The
validators check fabrication, not outcomes. Human grades record answer
shape for tracking, not pipeline requirements.

## No-live-spend rules

- Mock runs are the default. They cannot spend quota.
- Live runs need `--live` or `ARGUS_EVAL_LIVE=1`, plus a server with keys.
- Check `--dry-run` output before any live invocation.
- Never add provider calls to the mock or grading paths. `eval_grade.py`
  reads local files only.

## Where truth lives (v0.6 judgment)

Offline truth stays in JSONL. Run rows in `runs/<ts>.jsonl`, human grades
in `runs/<ts>.grades.jsonl`, baselines versioned under `evals/baselines/`.
The Grafana dashboard deliberately has no live verdict join. Per-question
Prometheus labels would blow up cardinality, and a grading script cannot
increment server counters, so the dashboard shows a text panel describing
this workflow plus an activity panel over existing low-cardinality metrics
(tool calls, loop stops, cost). Verdicts are read from the grades files,
not from Prometheus. If P6 needs live verdicts in dashboards, the honest
route is a pushgateway or an authed ingest endpoint, not per-case labels.
