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

## RAG Web App

- Location/reachability TBD. Same discovery treatment before any view code:
  base URL, document list/detail endpoints, auth shape, pagination.
