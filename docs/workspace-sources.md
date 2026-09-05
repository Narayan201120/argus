# Workspace sources (P4-4 discovery, verified live 2026-09-05)

## Research Radar (SaralAI_Assessment)

- Base URL: `http://127.0.0.1:8000` (docker compose; backend :8000, frontend :3000, postgres :5432)
- Auth: optional `X-API-Key` header; open when `API_KEY` unset (currently open)
- Rate limit: 60/min per IP on `/papers*` (429 + Retry-After)
- NEVER edit that repo from ARGUS work. Read-only + compose lifecycle only.

### GET /papers

Params: `q` (max 200 chars), `year`, `topic` (slug), `author` (substring),
`page` (default 1), `page_size` (1..100, default 20), `ranked` (BM25, needs `q`),
`hybrid` (RRF, needs `q`), `ids` (comma list, max 100; saved-papers flow).

Response: `{items, total, page, page_size}` where item =
`{id, title, publication_year, cited_by_count, authors: [{id, name}]}`.
NO abstract, NO url, NO score in list items.

### GET /papers/{id}

Response: `{id, title, abstract|null, publication_year, doi|null,
cited_by_count, created_at, authors[], topics: [{id, name, slug}]}`.
NO url field. Client builds links as `https://doi.org/{doi}`.

### GET /papers/{id}/similar

Response: `[{id, title, similarity_score}]`, fixed 5, cosine 4dp.

### Saved papers

No backend endpoint. Radar's own frontend keeps bookmarks in localStorage
and re-queries `GET /papers?ids=...`. ARGUS mirrors that: localStorage IDs,
no server state.

## RAG Web App (verified from source 2026-09-05, not yet running)

- Location: `C:\Users\naray\OneDrive\Desktop\Projects\rag_web_app` (NEVER edit; read-only + run only)
- Runs without docker: `backend/ python manage.py runserver [port]` (default :8000, conflicts; use :8001+), SQLite fallback, no venv present (owner starts it)
- API prefix `/api`. Auth: Django SimpleJWT, `Authorization: Bearer <access>` (30-min lifetime), refresh via cookie or `refresh` body field. Sign-in `POST /api/sign-in/` `{username, password}` → `{message, tokens:{access}}` + refresh cookie. No service credential concept: the ARGUS service identity is a real Django user seeing only its own corpus (per-user dirs + per-user FAISS index).
- Throttles: anon 5/min, user 30/min. Server-to-server needs only the Bearer header (+ refresh flow when 401).
- Mapping corrections applied to `app/tools/rag.py`: nested `tokens.access`, hit `source` field first, token TTL default 1500s.

### GET /api/documents/ (auth)

`{count, documents: [{name, size_bytes}]}`. Full array, no pagination.

### GET /api/documents/<filename>/ (auth)

`{name, extension, content (first 20000 chars), total_characters, truncated}`.

### GET /api/collections/ and GET /api/collections/<id>/ (auth)

List `{count, collections: [{id, name, description, document_count, created_at}]}`;
detail adds `documents: [{name, size_bytes}]`. No PUT/DELETE on collections.

### POST /api/search/ (auth)

Body `{query, top_k=3}` → `{query, count, results: [{chunk, source}]}`. No scores.
Rerank variant `POST /api/search/rerank/` adds `relevance_score` (not used v1).

### Writes (NOT proxied, RAG app only)

Upload, ingest, delete/rename/move documents, chat, feedback, tasks, admin, settings.
ARGUS Library is read-only: documents, preview, collections, search stays in
the investigation tool path.
