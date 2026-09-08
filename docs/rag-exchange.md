# RAG exchange contract (P7-2 Option C, primary)

Status: contract only. The RAG repo owner builds the endpoint. ARGUS side names are locked so the sibling workstream can land without drift.

Live-verified base this builds on: RAG sign-in is `POST /api/sign-in/` with `{username, password}`, returns `{message, tokens: {access}}` plus refresh cookie. Access lifetime is 30 min. ARGUS caches with a 1500s TTL and refreshes early. See `docs/workspace-sources.md`.

## Endpoint

`POST {rag}/api/service/exchange/` with body `{"assertion": "<HS256 JWT>"}`.

`{rag}` is `rag_base_url` with no trailing slash. `RAG_EXCHANGE_DEFAULT_PATH` is `/api/service/exchange/`. Setting `rag_exchange_url` overrides the full URL. `rag_exchange_enabled` gates the ARGUS caller. The shared HMAC secret is `RAG_EXCHANGE_SECRET` on both sides (`rag_exchange_secret` in ARGUS settings).

## Assertion claims

Minted by `mint_exchange_assertion(subject)` in `app/tools/rag.py`. Lifetime on our side is exactly 60s.

| Claim | Value | Notes |
| --- | --- | --- |
| `sub` | ARGUS subject | From `resolve_subject`. `"local"` in single-user mode, else the JWT `sub`. |
| `iss` | `JWT_ISSUER` or `"argus"` | Falls back to `"argus"` when `jwt_issuer` is unset. |
| `iat` | issued-at, seconds | UTC now. |
| `exp` | `iat + 60` | `exp - iat` is exactly 60 on our side. |

Signed HS256 with `RAG_EXCHANGE_SECRET`. No `kid` needed. RAG must reject `alg: none` and any unknown `kid`.

## Verification rules (RAG side)

1. Verify HS256 signature against `RAG_EXCHANGE_SECRET`.
2. Reject expired (`exp` past) and not-yet-valid (`iat` future beyond skew).
3. Reject `exp - iat > 120`. Our mint uses 60. The 120 cap leaves room for transit without accepting long-lived assertions.
4. Accept clock skew up to 30s on `iat`/`exp` checks. Same tolerance ARGUS already uses for its token cache.
5. `sub` must map to a Django user (next section). Unknown subject is 401, not 500.

## Success response

Same shape as the standard sign-in so the ARGUS caller reuses its token path:

```json
{"message": "ok", "tokens": {"access": "<bearer>"}, "expires_in": 1800}
```

`tokens.access` is required. `expires_in` is optional and defaults to 1500s on our side when absent. Standard TTL is 30 min.

## Errors

| Status | Meaning |
| --- | --- |
| 400 | Bad assertion. Bad signature, wrong alg, malformed body, lifetime over 120s, expired. |
| 401 | Valid assertion but unknown subject or no user mapping. |
| 403 | Exchange disabled on the RAG side. |

ARGUS maps these to a provider error for the caller. A 5xx from exchange is also a provider error.

## User mapping

Recommended: explicit table from `argus_sub` to Django username. The RAG owner keeps the table next to the exchange view so each ARGUS subject sees only its own corpus (per-user dirs and per-user FAISS index stay as they are).

Auto-provisioning new Django users on first exchange is explicitly undecided. The RAG owner must choose before rollout. Do not assume it exists.

## Throttle and logging

Exchanged tokens are normal per-user tokens. The 30/min per-user throttle applies to them. The 5/min anon throttle does not apply once a Bearer is attached.

Never log tokens or assertions. No assertion in logs, error bodies, or exception messages. Log only `sub`, `iss`, status code, and latency.

## Rollout and rollback

Flag off (`rag_exchange_enabled=false`) means legacy single-user sign-in with `rag_service_user` and `rag_service_pass`. That path stays until the owner confirms the exchange endpoint and mapping.

Exchange 5xx or timeout means ARGUS surfaces a provider error. No silent fallback to the shared-corpus service user. Falling back would mix corpora across users.

Order: ship the contract and the ARGUS mint behind the flag, confirm the RAG endpoint against staging, share the secret out of band, rotate the legacy service password after cutover.

## Owner-gated checklist

- RAG endpoint built at `POST /api/service/exchange/` with the verification rules above?
- `argus_sub` to Django user mapping chosen, explicit table in place?
- Auto-provision decision recorded (yes with rules, or no)?
- `RAG_EXCHANGE_SECRET` shared out of band, never in chat or tickets?
- Legacy service credentials rotated after cutover?
