# ARGUS Project Map

Last updated: August 23, 2026 (v0.3.0)

## Current Stage

ARGUS is a backend-only multi-model orchestration platform at **v0.3.0**.
Phase 1 (bounded parallel pipeline) and Phase 2 (intelligence, voice,
observability, release engineering) are complete and CI-guarded.

What is implemented now:
- FastAPI app: `POST /v1/query`, `POST /v1/query/stream` (SSE),
  `POST /v1/transcribe`, `POST /v1/query/audio`, `POST /v1/report`,
  `GET /v1/report/{id}`, `GET /v1/health`, `GET /v1/models`,
  `GET /v1/metrics`, `POST /v1/auth/token`
- Four connectors (Gemini, OpenAI, Claude, Mistral); model IDs
  env-configurable (`GEMINI_MODEL` / `MISTRAL_MODEL`); provider 429s
  surface as `rate_limited` with retry hints
- Router strategies: `static` (YAML chains) and `semantic`
  (embeddings-first via Gemini/OpenAI embeddings, keyword fallback,
  failure cooldown); profiles live in `config/routing.yaml` with
  per-profile `keywords:` and `description:`
- Deep-report mode: planner -> bounded parallel tracks -> global
  verifier -> writer -> reviewer repair loop, job store memory+Redis
- Sarvam speech-to-text input (saaras:v3) with credit-safety guards
- Redis response cache + rate limiting (subject-keyed when authed,
  IP fallback); everything fails open without Redis
- Opt-in JWT auth (client-credentials dev flow)
- Prometheus metrics + provisioned Grafana dashboard
  (`docker compose --profile observability up`) and opt-in OTel tracing
- CI gate on every push/PR: ruff, mypy, compileall, mock-only pytest

Not implemented (Phase 3 backlog):
- Conversation memory (Redis sessions)
- Local model connector (Ollama/LM Studio)
- Plugin SDK, A/B routing experiments
- Web UI (mic button), Kubernetes Helm chart

## Top-Level Structure

`app/` - application code (see Code Map)

`tests/` - pytest suite, 139 tests, mock-only (no live provider calls;
conftest forces embedding provider off in tests)

`scripts/smoke_live.py` - env-gated live smoke checks against a running
deployment (health/models/query/stream/report/audio)

`config/routing.yaml` - role preference chains + routing profiles
(connectors, keywords, description per profile)

`prompts/` - role prompt templates

`deploy/prometheus|grafana/` - observability stack configs + dashboard

`.github/workflows/ci.yml` - CI gate workflow

`README.md` - overview/setup | `PROJECT_MAP.md` - this map |
`PROGRESS.md` - progress log (local only) |
`DECISIONS.md` - decision log (local only) |
`ARGUS_PRD_v1.md` - PRD with architecture appendix (local only)

## Runtime Flow

`POST /v1/query`:
1. Async resolver validates model_config (unknown roles/connectors/
   strategies/profiles -> 422; nothing available -> 503) and applies the
   router strategy: explicit connectors > profile > semantic inference
   (embeddings -> keyword fallback) > full pool
2. `_is_simple_query` short-circuits simple queries to one direct
   provider call (with timeout-retry); cache consulted first when Redis
   is up (`cache_hit: true` on repeats)
3. Complex queries: `build_parallel_plan` freezes a `SharedTaskState`,
   researcher/analyzer/verifier run concurrently, each result parsed into
   typed contracts with real latency + token usage
4. `synthesize` reconciles role outputs through a synthesizer fallback
   chain into the final answer; successful responses cached

Other paths reuse the same resolver: `/v1/query/stream` emits SSE events
per role plus streamed synthesis tokens and a terminal `final` envelope;
`/v1/query/audio` transcribes via Sarvam then enters this pipeline;
`/v1/report` fans subtasks out through planner/tracks/verifier/writer/
reviewer asynchronously.

## Code Map

### Entry Point

`app/main.py`
- FastAPI app, lifespan: tracing setup -> connector registration ->
  Redis connect/close
- Middleware stack (outermost first): Prometheus -> JWT auth -> rate limit
- Routers under `/v1`: auth, audio, query, stream, reports, health,
  models, metrics

### Configuration

`app/config.py` - pydantic-settings: provider keys + model overrides,
Redis/cache/rate-limit, report rounds, Sarvam STT, embedding router,
JWT, tracing knobs

`pyproject.toml` - ruff/mypy/pytest config; version = 0.3.0

### API Layer

`app/api/schemas.py` - request/response models (incl. profile,
role_bindings, router_strategy on ConnectorConfigRequest; QueryResponse
carries role_assignments, cache_hit, router_strategy, matched_profile;
AudioQueryResponse adds transcript fields)

`app/api/routes/shared.py` - async `resolve_request_connectors` returning
frozen `ResolvedRouting(active, overrides, router_strategy,
matched_profile)`; single canonical 422/503 path for query/stream/report

`app/api/routes/query.py` - main orchestration endpoint + cache +
direct-path retry + role/token metric recording via `_build_status`

`app/api/routes/stream.py` - SSE variant (role_complete events, synthesis
tokens, terminal final envelope)

`app/api/routes/reports.py` - deep-report create/poll

`app/api/routes/audio.py` - transcribe + voice-query endpoints with
credit-safety guards (extension allowlist, size cap, no STT retries)

`app/api/routes/auth.py` - JWT issuance (dev client-credentials)

`app/auth.py` - JWT middleware; exempts /, docs, openapi.json, health,
auth/token, metrics

### Orchestration

`app/orchestration/contracts.py` - typed task/result/state contracts with
normalizing validators

`app/orchestration/decomposer.py` - short-query heuristic +
`build_parallel_plan`

`app/orchestration/workers.py` - `_query_with_retry` (timeout retry),
prompt assembly, JSON parsing, generic `WorkerOutcome[ResultT]`, public
run_*_task helpers

`app/orchestration/aggregator.py` - reconciliation + `synthesize` /
`synthesize_stream` fallback chains

`app/orchestration/binding.py` - `RoutingConfig` (rich YAML profiles =
ProfileDefinition{connectors, keywords, description}),
`RoleBindingService` (chains/profiles/keyword inference), ROUTER_STRATEGIES,
and `SemanticRouter` (embeddings-first classification, 60s cooldown,
keyword fallback) exposed as module singleton

`app/embeddings.py` - BaseEmbedder + OpenAI/Gemini backends + cosine +
factory (auto prefers Gemini)

`app/orchestration/report_{contracts,planner,jobs,runner}.py` -
deep-report contracts, LLM planner w/ fallback, memory+Redis job store,
pipeline executor with terminal-state metrics

### Connectors & Infrastructure

`app/connectors/base.py` - BaseConnector ABC, statuses incl.
RATE_LIMITED, `classify_provider_exception` (429/quota detection),
ConnectorResponse.retry_after_s, default stream_query delegating to query

`app/connectors/{gemini,openai,mistral}.py` + claude.py - providers;
SDK exceptions classified in handlers; model IDs from settings

`app/connectors/registry.py` - singleton registry; wraps instances with
OTel spans at registration when tracing enabled

`app/rediskit.py` - shared async Redis holder (fail-open everywhere)

`app/cache.py` - ResponseCache (sha256 keys, TTL, max-bytes guard)

`app/ratelimit.py` - fixed-window limiter; subject-keyed when
authenticated, IP otherwise; health/metrics exempt

`app/metrics.py` - argus_* metric registry + PrometheusMiddleware
(outermost; route-template labels resolved across starlette versions)

`app/tracing.py` - opt-in OTel configure_tracing() + span() context
manager

`app/transcription/` - BaseTranscriber, TranscriptionResult/Error,
SarvamTranscriber (api-subscription-key header, saaras:v3), factory

## Tests

139 passing (`venv\Scripts\python.exe -m pytest -q`), verified in both
the local venv and a fresh-resolution venv (CI parity). CI runs the same
suite on ubuntu/py3.12 plus ruff, mypy (app+scripts), compileall.

Coverage highlights: connector classification (quota->rate_limited),
router strategies (embedding match/threshold/fallback/cooldown via
scripted embedders), audio guards and voice-query pipeline (stubbed
transcriber), SSE event sequences, report lifecycle, auth flows, metric
label stability across starlette majors.

Gaps: no live-provider integration tests by design (DEC-008);
`scripts/smoke_live.py --live` covers that manually.

## Known Drift / Risks

1. Reconciliation heuristics are lexical, not semantic.
2. Parallel-path ModelStatus collapses non-success worker outcomes to
   `error` (worker raises); direct/stream paths preserve full status.
3. Dependency caps in requirements.txt are tested majors - bump
   deliberately (starlette 1.x lesson, DEC-043).
4. `google.generativeai` is deprecated upstream; migration to
   `google.genai` is an open candidate.
5. Semantic embeddings call providers per semantic query (cooldown after
   failures); Gemini free-tier quota applies.

## Roadmap Alignment

Phase 1 (Stages 0-8): COMPLETE. Phase 2 (P2-0..P2-4): COMPLETE at v0.3.0.
Next up when scheduled (Phase 3 backlog): conversation memory, Ollama/LM
Studio connector, plugin SDK, A/B routing experiments, web UI (mic
button), Kubernetes Helm chart.
