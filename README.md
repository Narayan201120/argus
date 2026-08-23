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

- `GET /` - service metadata
- `GET /v1/health` - live connector availability + Redis status
- `GET /v1/models` - registered connector profiles
- `POST /v1/query` - direct or parallel orchestration
- `POST /v1/query/stream` - SSE variant: role completion events, streamed synthesis tokens, terminal `final` envelope
- `POST /v1/report` - start a deep-report job (returns job ID immediately)
- `GET /v1/report/{job_id}` - poll status; completed jobs carry Markdown

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
- `semantic` - when the caller set no explicit profile, a keyword intent classifier infers one of the named profiles from the query text (`research`, `code`, `analysis`, `fast`). An explicit `model_config.profile` or `model_config.connectors` always wins over inference.

Responses expose `router_strategy` and `matched_profile` so callers can see which path served the request.

## Testing

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## Current Scope

This is a backend-only MVP. The bounded parallel `/v1/query` pipeline remains the fast default path; `POST /v1/report` adds the deep-report mode (planner -> parallel research tracks -> global verification -> writer -> bounded reviewer repair) returning Markdown asynchronously. Streaming (SSE), JWT auth, and metrics endpoints are the next milestones.

## Security

Use `.env.template` only as a blank template. Keep secrets in `.env`, rotate any key that has ever been committed or exposed, and never commit provider credentials.
