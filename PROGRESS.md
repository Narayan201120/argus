# Project ARGUS — Progress Log

Last updated: March 6, 2026

---

## Snapshot

- **Project type:** Multimodal AI Orchestration Platform — Python (FastAPI) backend + React frontend (Phase 3)
- **Status:** Pre-implementation — design & architecture phase complete
- **Phase:** Phase 1 (Foundation / MVP) — not yet started
- **Stack decided:** Python 3.11+, FastAPI, asyncio, httpx, Redis, Docker Compose
- **MVP Connectors:** Google Gemini, OpenAI GPT-4o, Anthropic Claude
- **Synthesizer:** Claude (primary) → GPT-4o → Gemini → labeled concat (fallback chain)

---

## Architecture Decisions Log

These design decisions were finalized before implementation began. Locked-in for MVP.

### Core Architecture

- ARGUS is a **5-layer cloud-native orchestration system:**
  - Layer 1: API Gateway (FastAPI — auth, rate limiting, input validation, multimodal parsing)
  - Layer 2: Decomposer / Query Planner (splits query into N sub-queries aligned to model capabilities)
  - Layer 3: Dispatcher (async fan-out via `asyncio.gather()` — sends sub-queries to connectors)
  - Layer 4: Model Connectors (`BaseConnector` interface — Gemini, GPT-4o, Claude)
  - Layer 5: Aggregator / Synthesizer (Claude — combines complementary sub-answers into one response)
  - Layer 6: Observability (Prometheus, structured JSON logging, OpenTelemetry)

### Query Decomposition (Key Architectural Decision)

- **Decomposer-first design:** Instead of sending the same query to all models (redundant), the Decomposer splits the main query into N meaningful sub-queries, where N = number of user-selected connectors.
- Each sub-query is assigned to the model best suited for that slice of the problem (Gemini → research, GPT → code, Claude → reasoning/analysis).
- **Short-circuit rule:** If the query is too simple (under threshold token count or single-intent), the Aggregator responds directly without invoking any connectors.
- Each connector receives: original query (as context) + assigned sub-query.

### Connector Selection

- Users choose which connectors to activate per query. No model is mandatory.
- Named profiles (`research`, `code`, `analysis`, `fast`) map to pre-configured connector sets.
- The synthesizer role is separate from connector roles (Claude can act as both connector and synthesizer but from separate config pools).

### Synthesizer Fallback Chain

```
Claude (primary)
  → GPT-4o (fallback #1)
  → Gemini (fallback #2)
  → Labeled concatenation (last resort — raw but functional)
```

- If the primary synthesizer fails or times out, the system automatically promotes the next in the chain.

### Streaming Design (3-Phase SSE)

- `/v1/query` — blocking, returns full synthesized JSON response
- `/v1/query/stream` — Server-Sent Events, 3 phases:
  - **Phase 1 (Planning):** Decomposer outputs sub-query assignments per model
  - **Phase 2 (Execution):** Each connector streams its answer to its sub-query in real time
  - **Phase 3 (Synthesis):** Claude streams the final unified response token by token
- `stream_mode: "synthesis_only"` skips Phase 1 & 2 SSE events — only streams final synthesis

### Graceful Degradation

- `asyncio.gather(return_exceptions=True)` — per-connector exception capture
- Failed connectors are logged, flagged in metadata, excluded from synthesis
- Timeout per connector: configurable, defaults to **45–60 seconds** (not 30s — accounts for real-world LLM + network latency)
- One automatic retry on timeout before marking connector degraded
- System proceeds with partial results if M < N connectors respond

### Cost Control

- User selects connectors — no forced "dispatch to all" by default
- Per-session spending caps enforced at API Gateway via Redis counters
- Per-connector token usage tracked in response metadata envelope

### Security

- Provider API keys stored in environment variables or secrets manager (never in code or logs)
- JWT auth on all API endpoints (Phase 1 MVP)
- Rate limiting per-user and global via Redis
- Input sanitization before dispatch

---

## What Has Been Completed

### Pre-Implementation (Design Phase)

- [x] PRD v1.0 authored (`ARGUS_PRD_v1.md`)
- [x] Architecture reviewed and critiqued
- [x] Query Decomposer design finalized (replaces same-query-to-all pattern)
- [x] Synthesizer fallback chain defined
- [x] Streaming architecture designed (3-phase SSE)
- [x] Cost control strategy agreed
- [x] Graceful degradation SLA revised (45–60s timeout, not 100ms)
- [x] Open design questions resolved (see Architecture Decisions Log above)

---

## What Is In Progress

- [ ] Nothing started yet — implementation begins next session

---

## Phase 1 — Foundation (MVP) Checklist

- [ ] FastAPI project scaffold with async architecture and Pydantic models
- [ ] `BaseConnector` abstract interface + `ConnectorRegistry` implementation
- [ ] Gemini 1.5+ connector (text + vision + PDF support)
- [ ] GPT-4o connector (text + vision + code)
- [ ] Claude connector (text + document + reasoning)
- [ ] `asyncio.gather()` parallel dispatcher with per-connector timeout + retry
- [ ] Decomposer / Query Planner (LLM-backed, lightweight model — Gemini Flash or GPT-4o-mini)
- [ ] Short-circuit logic (single-intent / short query → Aggregator responds directly)
- [ ] Aggregator with synthesizer fallback chain (Claude → GPT-4o → Gemini → concat)
- [ ] Synthesis prompt v1 (baseline — will evolve; versioned in `prompts/`)
- [ ] `POST /v1/query` endpoint with `model_config` toggle support
- [ ] Rule-based Dispatcher (modality + keyword routing as fallback)
- [ ] Redis integration (response caching + rate limit counters + spending caps)
- [ ] Docker Compose dev environment
- [ ] pytest suite with connector mocks and integration tests
- [ ] `GET /v1/health` — per-connector status + degradation flags
- [ ] Graceful degradation: partial response on connector failure

## Phase 2 — Intelligence & Streaming Checklist

- [ ] `POST /v1/query/stream` — Server-Sent Events 3-phase streaming
- [ ] Semantic Router (embedding-based intent classifier — FAISS + lightweight encoder)
- [ ] Named routing profiles (`research`, `code`, `analysis`, `fast`) via YAML config
- [ ] Audio input support: Whisper / Gemini Audio transcription pipeline
- [ ] Prometheus metrics endpoint + Grafana dashboard template
- [ ] Decomposer model upgrade / tuning (Phase 2 intelligence pass)
- [ ] Per-query cost estimate (optional, pre-dispatch)

## Phase 3 — Scale & Extensibility Checklist

- [ ] Local model connector: Ollama / LM Studio integration
- [ ] Plugin SDK: documented interface + CLI for external connector development
- [ ] Conversation memory: multi-turn context via Redis session store
- [ ] A/B routing experiments: traffic splitting with quality metrics
- [ ] Web UI: React interface with model toggles, query history, response diff view
- [ ] Kubernetes Helm chart for production deployment

---

## Current Known Issues & Blockers

- None — project has not started implementation yet.

---

## Persisting Design Questions (To Resolve During Implementation)

- **Decomposer model choice:** Gemini Flash vs GPT-4o-mini — decide based on latency benchmarks during Phase 1 development
- **Short-circuit threshold:** Token count cutoff for direct Aggregator response (to be tuned empirically)
- **Sub-query visibility UX:** Whether to expose sub-query assignments to users in Phase 1 or Phase 2
- **Synthesis prompt:** Will evolve; versions tracked in `prompts/` directory
- **Synthesizer configurability:** Always Claude for MVP; user-selectable in Phase 2

---

## Notes For New Sessions

- This file is the primary handoff document. Always read this before starting a new session.
- PRD is at `ARGUS_PRD_v1.md` — the canonical product spec (note: architecture decisions above supersede some sections of PRD v1.0 which predates these decisions).
- Stack: Python 3.11+, FastAPI, asyncio, httpx, Redis, Docker Compose.
- Connector interface pattern: `BaseConnector.query(prompt, config) -> ConnectorResponse` — every model adapter implements this.
- Synthesis prompts are versioned files in `prompts/` — never hardcoded inline.
- All provider API keys go in `.env` only — never in source code.
- Use `asyncio.gather(return_exceptions=True)` for all connector fan-out — never bare gather.
