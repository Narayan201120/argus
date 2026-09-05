# ARGUS Project Map

Last updated: September 5, 2026 (v0.5.0 - Phase 4 complete)

## Current Stage

ARGUS is a deep-research orchestration platform at **v0.5.0**.
Phase 1 (bounded parallel pipeline), Phase 2 (intelligence, voice,
observability), and Phase 3 (web UI, memory, A/B experiments, SDK) are
complete and CI-guarded. Phase 4 (DEC-053) adds the investigation loop:
`POST /v1/investigate` fans out parallel tool calls (Radar, RAG, web)
onto an Evidence Board (Investigation/Evidence/Claim), concurrent
analysis/critique/gap workers steer adaptive follow-up rounds under
config budgets (iterations, tool calls, wall clock, cost), and milestone
synthesis streams a Markdown report to completion.

What is implemented now:
- FastAPI app: `POST /v1/query`, `POST /v1/query/stream` (SSE),
  `POST /v1/transcribe`, `POST /v1/query/audio`, `POST /v1/speak`,
  `POST /v1/report`, `GET /v1/report/{id}`, `POST /v1/investigate`,
  `GET /v1/investigate/{id}`, `POST /v1/investigate/{id}/cancel`,
  `POST /v1/investigate/{id}/feedback`, `GET /v1/investigate/{id}/stream`,
  `GET /v1/investigations`, `GET /v1/radar/papers*` proxies,
  `GET /v1/library/*` proxies, `GET /v1/health`, `GET /v1/models`,
  `GET /v1/metrics`, `POST /v1/auth/token`, `POST /v1/feedback`
- Four connectors (Gemini, OpenAI, Claude, Mistral); model IDs
  env-configurable (`GEMINI_MODEL` / `MISTRAL_MODEL`); provider 429s
  surface as `rate_limited` with retry hints; repeated auth failures
  demote a connector until a success restores it
- Router strategies: `static` (YAML chains) and `semantic`
  (embeddings-first via Gemini/OpenAI embeddings, keyword fallback,
  failure cooldown); profiles live in `config/routing.yaml` with
  per-profile `keywords:` and `description:`
- Deep-report mode: planner -> bounded parallel tracks -> global
  verifier -> writer -> reviewer repair loop, job store memory+Redis
- Investigation loop: `app/tools/` (radar/rag/web + registry + dispatch),
  `app/evidence/` (board models + snapshot store), `app/analysis/`
  (board renderer, workers, loop, milestone synthesis, event bus);
  budgets `INVESTIGATION_MAX_ITERATIONS/MAX_TOOL_CALLS/MAX_WALL_TIME_S/
  MAX_WEB_CALLS/MAX_COST_USD`; terminal reasons incl `COST_LIMIT`
- Web tools: Tavily search + trafilatura fetch (`TAVILY_API_KEY`,
  follow-up rounds only, SSRF guard, size caps)
- Workspace UI (React+Bun, served from `/`): chat, investigations with
  live board + streaming report, Radar search/saved, RAG library,
  investigation history, chat-to-investigation handoff
- Sarvam speech-to-text input (saaras:v3) with credit-safety guards
- Redis response cache + rate limiting (subject-keyed when authed,
  IP fallback); everything fails open without Redis
- Opt-in JWT auth (client-credentials dev flow)
- Prometheus metrics + provisioned Grafana dashboard (investigation
  times, loop stops, spend, feedback)
  (`docker compose --profile observability up`) and opt-in OTel tracing
- CI gate on every push/PR: ruff, mypy, compileall, mock-only pytest

Not implemented (future scope):
- Local model connector (Ollama/LM Studio) - deferred, no local compute
- Per-user RAG authorization mapping (service identity is temporary,
  RAG stays final authority)
- OAuth2
- Kubernetes Helm chart

In progress: nothing open. Phase 4 complete; live smoke of one
end-to-end investigation waits on working provider keys.

## Top-Level Structure

`app/` - application code (see Code Map)

`tests/` - pytest suite, 296+ tests, mock-only (no live provider calls;
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

`pyproject.toml` - ruff/mypy/pytest config; version = 0.5.0

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

`app/api/routes/sessions.py` - GET/DELETE working-memory sessions

`app/api/routes/feedback.py` - POST/GET quality ratings for A/B routing

`app/memory.py` - SessionStore: Redis rolling Q/A turns per session_id,
fail-open; bounded transcript builder for prompt injection

`app/feedback.py` - rating persistence (Redis, fail-open)

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
Phase 3 (P3-0..P3-7, DEC-048): COMPLETE at v0.4.0 - debt/polish, React UI,
voice loop, working memory, A/B experiments + feedback, connector SDK.
Phase 4 (P4-0..P4-5, DEC-053): COMPLETE at v0.5.0 - Evidence Board,
tool layer (Radar/RAG/web), analysis workers + adaptive loop, milestone
streaming synthesis, workspace UI, cost governance, dashboards.
Deferred to future scope: local models (no compute), Kubernetes/Helm
(revisit when a deployment target exists), long-term memory recall,
per-user RAG authorization mapping, response diff view, streaming TTS.
