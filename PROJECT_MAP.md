# ARGUS Project Map

Last updated: August 22, 2026

## Current Stage

ARGUS is a backend-only Phase 1 MVP for multi-model orchestration using a
bounded role-based parallel pipeline.

What is implemented now:
- FastAPI app with `POST /v1/query`, `GET /v1/health`, and `GET /v1/models`
- Connector abstraction and registry; all four connectors (Gemini, OpenAI,
  Claude, Mistral) registered at startup and gated by API key presence
- Deterministic parallel plan builder (`build_parallel_plan`) with a frozen
  shared task snapshot
- Parallel role workers (`researcher` / `analyzer` / `verifier`) with
  retry-on-timeout and real token usage propagation (`WorkerOutcome`)
- Aggregator with role precedence, deterministic reconciliation summary, a
  synthesizer fallback chain, and labeled-concat last resort
- Short-query direct path that bypasses the pipeline
- Tooling: ruff + mypy configured via `pyproject.toml`, both clean

What is not implemented yet:
- Redis-backed caching or rate limiting in business logic (Redis is
  provisioned in Docker Compose but unused)
- Streaming endpoint
- Frontend
- End-to-end validation with real provider keys

## Top-Level Structure

`app/`
- Main application code

`tests/`
- Unit and API tests (pytest + pytest-asyncio, mock connectors)

`prompts/`
- Role prompt templates consumed by orchestration logic

`README.md`
- High-level project overview and setup

`PROGRESS.md`
- Handoff/progress log (kept current after each session)

`ARGUS_PRD_v1.md`
- Original product requirements document (local only)

## Runtime Flow

Current request flow for `POST /v1/query`:

1. API resolves requested vs available connectors (unknown IDs -> 422,
   none available -> 503)
2. `_is_simple_query` short-circuits short single-intent queries to one
   direct provider call
3. Otherwise `build_parallel_plan` produces an `OrchestrationPlan`
   (`SharedTaskState` + per-role tasks)
4. Researcher, analyzer, and verifier run concurrently via
   `asyncio.gather(..., return_exceptions=True)` in the route
5. `_build_status` converts each `WorkerOutcome` (or exception) into a
   `ConnectorResponse` carrying real latency and token usage
6. `synthesize` reconciles parsed role outputs plus a deterministic
   reconciliation summary into the final answer
7. Response includes per-role statuses, synthesizer ID, and latency
   breakdown

## Code Map

### Entry Point

`app/main.py`
- Creates the FastAPI app
- Registers all four connectors during lifespan startup
- Mounts API routers under `/v1`

### Configuration

`app/config.py`
- Loads `.env` values via `pydantic-settings`
- Holds provider keys, Redis URL, connector defaults

`pyproject.toml`
- Ruff lint config (app strict; tests exempt from line-length for JSON
  fixtures), mypy config, pytest config

### API Layer

`app/api/schemas.py`
- Pydantic request/response models (`model_config` alias handling)

`app/api/routes/query.py`
- Main orchestration endpoint
- `ROLE_PREFERENCES` maps roles to provider preference chains
- Builds active connector list, runs plan/workers/synthesis

`app/api/routes/health.py`
- Live connector availability checks

`app/api/routes/models.py`
- Lists registered connectors and capabilities

### Orchestration Layer

`app/orchestration/contracts.py`
- Typed Pydantic contracts: shared task state, per-role tasks and results,
  aggregation input, orchestration plan
- Field validators normalize whitespace and deduplicate lists

`app/orchestration/decomposer.py`
- `_is_simple_query` heuristic (word count, question marks, newlines)
- `build_parallel_plan` constructs the frozen snapshot and role tasks

`app/orchestration/workers.py`
- `_query_with_retry`: one automatic retry on timeout
- `_run_role_task`: prompt assembly from templates, JSON parsing into
  typed results, returns `WorkerOutcome(result, response)`
- Public entry points: `run_research_task`, `run_analysis_task`,
  `run_verification_task`

`app/orchestration/aggregator.py`
- Parses role outputs into typed models
- Deterministic reconciliation: unsupported assumptions, missing validation
  coverage, constraint/risk conflicts, confidence scoring (token-overlap
  heuristics - acceptable for MVP, not semantic ground truth)
- Synthesis fallback chain then labeled concatenation

Note: the former `dispatcher.py` and legacy LLM-based `decompose_query`
path were removed on 2026-08-22; fan-out now lives in the query route and
retry logic lives in `workers._query_with_retry`.

`app/orchestration/binding.py`
- `RoutingConfig`, YAML loader with default fallback, and
  `RoleBindingService.select_connector`
- Loads `config/routing.yaml` (roles + named profiles)

`app/cache.py`
- `ResponseCache`: sha256(query+model_config) keys, TTL, fail-open reads

`app/ratelimit.py`
- `RateLimitMiddleware`: fixed-window Redis counters per client IP,
  429 + Retry-After, fails open without Redis

`app/rediskit.py`
- Shared async Redis client holder, connect/close/ping helpers

`app/orchestration/report_contracts.py`
- Deep-report models: `ReportPlan`, `ReportSubtask`, `TrackResult`,
  `ReviewVerdict`, `VerificationSummary`

`app/orchestration/report_planner.py`
- LLM planner splitting a request into 2-5 subtasks; deterministic
  single-subtask fallback on any planner/parse failure

`app/orchestration/report_jobs.py`
- `ReportJobStore`: in-memory jobs mirrored to Redis (24h TTL), with
  read-through restore and corrupt-payload protection

`app/orchestration/report_runner.py`
- Pipeline executor: bounded parallel tracks (semaphore 3) reusing role
  workers, global verifier pass, writer, one-round reviewer repair loop,
  labeled Markdown fallback

`app/api/routes/reports.py`
- `POST /v1/report` (202 + job id, fire-and-forget task) and
  `GET /v1/report/{id}`; validates profiles/bindings/connectors like the
  query route

### Connector Layer

`app/connectors/base.py`
- `BaseConnector` ABC, `ConnectorStatus` StrEnum, response/config/usage
  dataclasses

`app/connectors/registry.py`
- Singleton registry populated at startup

`app/connectors/{gemini,openai,claude,mistral}.py`
- Provider implementations; availability = API key present; real token
  usage extracted where SDKs expose it

### Utilities

`app/utils/logger.py`
- JSON structured logging

## Tests

Suite: 33 passing locally (`venv\Scripts\python.exe -m pytest -q`),
ruff and mypy clean on `app/`.

Coverage focus:
- connector abstraction basics
- simple-query heuristic and parallel plan builder
- worker prompt building, JSON parsing, timeout-retry behavior
- aggregator reconciliation and fallbacks
- API routes: direct path, requested-provider isolation, registration

Gaps:
- no real provider integration tests
- no end-to-end tests with actual API keys

## Current Architectural Constraints

The system is a bounded parallel pipeline, not a dynamic multi-agent graph:
- planning happens once (deterministically)
- worker execution happens in parallel
- synthesis happens once

## Known Drift / Risks

1. Reconciliation heuristics are lexical (token overlap), not semantic.
2. No live-provider validation has been performed (keys pending rotation).
3. Named profile provider orderings in `config/routing.yaml` are starting
   points, not benchmarked choices.

## Roadmap Alignment

See `PROGRESS.md`. Completed: Stage 0 (observability + dead code removal),
Stage 1 (routing config + binding), Stage 2 (Redis cache + rate limiting),
Stage 3 (deep-report mode). Next: SSE streaming, JWT auth, semantic
router, metrics, release hardening.
