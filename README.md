# ARGUS

**Agentic Retrieval & Graph-based Understanding System**  
*A Multimodal AI Orchestration Platform*

> Private repository — CONFIDENTIAL

---

## What Is ARGUS?

ARGUS is an AI orchestration platform that accepts a user query, decomposes it into specialized sub-queries aligned to each model's strengths, fans them out in parallel across frontier AI models (Gemini, GPT-4o, Claude), and synthesizes a single authoritative response.

Rather than every model answering the same question redundantly, each model owns a focused slice of the problem. The synthesizer combines complementary outputs into one coherent answer.

## Architecture at a Glance

```
User Query
    │
    ▼
Decomposer / Query Planner      ← Splits query into N sub-queries per model capability
    │
    ▼
Parallel Fan-out (asyncio)      ← Gemini | GPT-4o | Claude — each gets its own sub-query
    │
    ▼
Aggregator / Synthesizer        ← Claude (with fallback chain) — merges into unified response
    │
    ▼
Structured Response             ← { result, metadata, model_statuses, latency_breakdown }
```

## Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.11+ |
| Web Framework | FastAPI |
| Async Execution | asyncio + httpx |
| Caching / Rate Limiting | Redis |
| Containerization | Docker Compose |
| Observability | Prometheus + OpenTelemetry |
| Testing | pytest + pytest-asyncio |

## Roadmap

- **Phase 1 (MVP):** Core orchestration — Decomposer, Dispatcher, Connectors, Aggregator, REST API
- **Phase 2:** Streaming (SSE), semantic routing, audio input, auth
- **Phase 3:** Local model support, Plugin SDK, Web UI, Kubernetes

## Docs

- [`ARGUS_PRD_v1.md`](./ARGUS_PRD_v1.md) — Full product requirements document
- [`PROGRESS.md`](./PROGRESS.md) — Live progress log, architecture decisions, and session handoffs
