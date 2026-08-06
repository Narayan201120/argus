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
- `GET /v1/health` - live connector availability
- `GET /v1/models` - registered connector profiles
- `POST /v1/query` - direct or parallel orchestration

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

Docker Compose starts the API and Redis. Redis is provisioned for the next caching/rate-limiting milestone but is not yet used by request handling.

## Testing

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

## Current Scope

This is a backend-only Phase 1 MVP. The next architectural extension is an optional deep-report mode that fans out independent research/analysis tracks by subtask, aggregates them, then writes Markdown. Reviewer-driven repair loops remain a follow-up rather than the default latency path.

## Security

Use `.env.template` only as a blank template. Keep secrets in `.env`, rotate any key that has ever been committed or exposed, and never commit provider credentials.
