# ARGUS

ARGUS is a FastAPI backend for fast, multi-model research and answer generation. It uses a bounded fan-out/fan-in pipeline: independent role workers run concurrently, then a synthesizer reconciles their structured outputs into one response.

## Current Architecture

```text
User query
  -> short query: one direct provider response
  -> long query: planner -> researcher + analyzer + verifier in parallel -> synthesizer
```

The parallel roles receive the same frozen task snapshot:

- `researcher`: facts, constraints, references, and unknowns
- `analyzer`: solution path, assumptions, tradeoffs, risks, and validation checks
- `verifier`: critical risks, hidden assumptions, edge cases, and validation requirements

The aggregation layer validates role JSON, applies role precedence, identifies unsupported assumptions and missing validation coverage, and asks the selected synthesizer for the final answer.

## Providers

ARGUS has connector implementations for Gemini, OpenAI, Claude, and Mistral. All four are registered at startup; a connector is usable only when its corresponding key is present in `.env`.

Provider selection is request-scoped. Send `model_config.connectors` to restrict the request to particular available providers, or omit it to use every available provider.

## Quick Start

1. Create a local environment file:

```powershell
Copy-Item .env.template .env
```

2. Put only active, rotated keys in `.env`.

3. Install dependencies and run the API:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

## API

- `GET /v1/meta` - service metadata
- `POST /v1/query/audio` - voice question: audio in, answer out (Sarvam STT)
- `POST /v1/speak` - text-to-speech: returns WAV audio (Sarvam Bulbul)
- `POST /v1/feedback` - rate a past answer 1-5 by `request_id` (A/B quality signal); `GET /v1/feedback/{id}` to read it back
- `GET /v1/session/{id}` / `DELETE /v1/session/{id}` - inspect or wipe a conversation's working memory
- `GET /v1/routing` - router strategies + named profiles (for UI pickers)
- `GET /v1/health` - live connector availability + Redis status
- `GET /v1/models` - registered connector profiles
- `POST /v1/query` - direct or parallel orchestration
- `POST /v1/query/stream` - SSE variant: role completion events, streamed synthesis tokens, terminal `final` envelope
- `POST /v1/report` - start a deep-report job (returns job ID immediately)
- `GET /v1/report/{job_id}` - poll status; completed jobs carry Markdown
- `POST /v1/investigate` - start a deep-research investigation (202 + ID immediately; Phase 4)
- `GET /v1/investigate/{id}` - full Evidence Board: investigation, evidence, claims, counts
- `POST /v1/investigate/{id}/cancel` - stop an investigation; board snapshot is kept
- `GET /v1/investigate/{id}/stream` - SSE: board snapshot then live round/evidence/claim/synthesis events to terminal

Example request:

```json
{
  "query": "Compare retrieval augmented generation with long-context prompting for an internal knowledge base.",
  "model_config": {
    "connectors": ["gemini", "mistral"],
    "timeout_s": 45,
    "max_tokens": 4096,
    "temperature": 0.3
  }
}
```

A short, single-intent query uses one selected provider directly. Longer or multi-part queries run the three role workers in parallel. Responses include per-role provider status and latency in `model_statuses`.

## Docker

```powershell
docker compose up --build
```

Docker Compose starts the API and Redis.

## Caching & Rate Limiting

Successful `/v1/query` responses are cached in Redis (TTL configurable via `CACHE_TTL_S`). Identical query + model_config pairs return a cached response with `cache_hit: true`. Requests are rate limited per client IP with a fixed window (`RATE_LIMIT_MAX_REQUESTS` per `RATE_LIMIT_WINDOW_S`; excess requests receive `429`). When Redis is unreachable the API degrades gracefully: caching and rate limiting disable themselves rather than failing requests, and `/v1/health` reports the Redis status.

## Routing Profiles & Role Binding

Role-to-provider preference chains and named profiles live in `config/routing.yaml`. Per request:

- omit both to use every available connector,
- set `model_config.profile` (e.g. `"fast"`, `"research"`), or
- set `model_config.connectors` explicitly (wins over profile).

Set `model_config.role_bindings` to override which provider fills `researcher`, `analyzer`, `verifier`, or `synthesizer`. Every response includes `role_assignments` showing the actual provider used per role.

### Router strategies

`ROUTER_STRATEGY` (default `"static"`) selects how the connector pool is chosen; a request can override it via `model_config.router_strategy` (unknown values return `422`):

- `static` - fixed YAML preference chains per role, exactly as configured in `config/routing.yaml`.
- `semantic` - when the caller set no explicit profile, the query is classified against profile descriptions: **embeddings-first** (Gemini/OpenAI embeddings, cosine similarity, `ROUTER_EMBEDDING_THRESHOLD`) with automatic fallback to a keyword classifier on weak matches or embedding failures (60s cooldown). An explicit `model_config.profile` or `model_config.connectors` always wins over inference.

Responses expose `router_strategy` and `matched_profile` so callers can see which path served the request; `argus_router_decisions_total{method,matched_profile}` tracks which mechanism decided.

### A/B experiments

Set `ROUTER_AB_SPLIT=semantic:80,static:20` and every request *without* an explicit strategy is deterministically assigned by hashing its question text (same question → same group). Rate answers with `POST /v1/feedback` and compare per-strategy satisfaction via `argus_feedback_total` in Grafana. Empty setting disables experiments.

## Observability

`GET /v1/metrics` exposes Prometheus-format metrics from the default registry: HTTP request counts and latency histograms per route template (`argus_http_requests_total`, `argus_http_request_duration_seconds`), in-flight gauge (`argus_http_in_flight`), cache hit/miss counters (`argus_cache_operations_total`), rate-limit rejections (`argus_rate_limit_rejections_total`), per-role worker outcomes/latency/token series (`argus_role_outcomes_total`, `argus_role_latency_seconds`, `argus_role_tokens_total`), report job acceptance (`argus_report_jobs_total`), router decisions (`argus_router_decisions_total`), and transcription stats (`argus_transcriptions_total`, `argus_transcription_latency_seconds`). The endpoint is exempt from both auth and rate limiting so scrapers never need tokens.

### Dashboards & tracing

A pre-built Grafana dashboard ships at `deploy/grafana/dashboards/argus.json`. Start the whole observability stack alongside the API:

```bash
docker compose --profile observability up -d
# Grafana: http://localhost:3000  (dashboard auto-provisioned)
# Prometheus: http://localhost:9090
```

Distributed tracing is **opt-in** OpenTelemetry: set `TRACING_ENABLED=true` and spans are emitted per connector call and per transcription (`TRACING_EXPORTER=console` by default; `otlp` needs `opentelemetry-exporter-otlp` installed plus `TRACING_OTLP_ENDPOINT`). Disabled = zero overhead.

## Testing

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

CI (`.github/workflows/ci.yml`) runs the full gate on every push to `main` and every PR: `ruff check .`, `mypy app scripts`, `compileall`, and the mock-only pytest suite - no API keys or Redis required.

### Live smoke test

`scripts/smoke_live.py` exercises a **running** deployment end-to-end (health, models, query, SSE stream, deep-report). It is env-gated: it refuses to run without `--live` or `ARGUS_SMOKE_LIVE=1`, because real runs spend provider tokens.

```powershell
# start the stack first, with real keys in .env
docker compose up --build -d

python scripts/smoke_live.py --live                                   # all checks
python scripts/smoke_live.py --live --only health models              # subset
python scripts/smoke_live.py --live --token <jwt>                     # auth-enabled deploy
```

Exit codes: `0` passed, `1` failures, `2` guard refused, `3` usage error.

## Current Scope

This is a backend-only MVP at **v0.4.0**. The bounded parallel `/v1/query` pipeline remains the fast default path; `POST /v1/report` adds the deep-report mode (planner -> parallel research tracks -> global verification -> writer -> bounded reviewer repair) returning Markdown asynchronously. Shipped alongside it: SSE streaming (`/v1/query/stream`), opt-in JWT auth, Redis caching/rate limiting, an embeddings-first semantic router (`static`/`semantic`, with A/B experiments via `ROUTER_AB_SPLIT` + `POST /v1/feedback`), **voice in and out** — Sarvam speech-to-text (`/v1/transcribe`, `/v1/query/audio`) and text-to-speech (`POST /v1/speak`) with a built-in web UI featuring a mic button, editable transcripts, spoken answers and per-answer ratings — **working-memory conversations** (`session_id`, Redis-backed, `GET/DELETE /v1/session/{id}`), a connector SDK (`docs/connectors.md` + `scripts/new_connector.py`), Prometheus metrics with a provisioned Grafana dashboard, and opt-in OpenTelemetry tracing. Deferred until needed: long-term memory recall, multimodal inputs beyond audio, OAuth2, local model connectors (Ollama), Kubernetes/Helm.

## Security

Use `.env.template` only as a blank template. Keep secrets in `.env`, rotate any key that has ever been committed or exposed, and never commit provider credentials.
