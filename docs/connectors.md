# Adding a New Provider Connector

Everything needed to wire a new AI provider into ARGUS in ~30 minutes.
Fastest path: run the scaffold, then fill the marked TODOs.

```powershell
python scripts/new_connector.py myprovider --display-name "MyProvider"
```

## 1. The contract

Subclass `BaseConnector` (`app/connectors/base.py`) and implement two
methods; everything else is provided or inherited.

```python
class MyProviderConnector(BaseConnector):
    connector_id = "myprovider"          # unique, lowercase
    display_name = "My Provider"
    capabilities = ["text"]              # advertised in /v1/models

    def __init__(self):
        self.api_key = settings.myprovider_api_key
        self.default_model = settings.myprovider_model
        self.is_available = bool(self.api_key)   # gated by key presence

    async def query(self, prompt, sub_query, config: ConnectorConfig) -> ConnectorResponse:
        ...

    async def health_check(self) -> bool:
        return bool(self.api_key)
```

Rules of the contract:

| Rule | Why |
|---|---|
| Never raise from `query()` | Return a `ConnectorResponse` with an error/status instead - raising breaks the response envelope |
| Classify exceptions with `classify_provider_exception(e)` | Maps HTTP 429/quota replies to `RATE_LIMITED` + captures Retry-After |
| Honor `config.timeout_s`, `.temperature`, `.max_tokens`, `.model_override` | Routes and users rely on them (`model_override` lets callers pin a specific model) |
| Report real token usage when the SDK exposes it | Feeds `/v1/query` responses, UI badges and Grafana |

`stream_query()` is inherited free: it delegates to `query()` and yields
one chunk. Override it only for native token streaming.

## 2. Response statuses you can return

| Status | When |
|---|---|
| `SUCCESS` | Content produced |
| `TIMEOUT` | Your own deadline exceeded (callers may fail over to another provider) |
| `ERROR` | Anything else |
| `RATE_LIMITED` | Provider replied 429/quota - include `retry_after_s` when known |

The direct path fails over down the preference chain on any non-success,
and parallel roles surface your true status to clients.

## 3. Configuration pattern

1. Add two fields in `app/config.py`:

```python
myprovider_api_key: str | None = None
myprovider_model: str = "whatever-default"
```

2. Add matching lines to `.env.template` (empty values only).

Model IDs are env-tunable on purpose (see `GEMINI_MODEL`) so provider
side deprecations never require a code change.

## 4. Registration

One line in `app/main.py` lifespan, alongside the others:

```python
registry.register(MyProviderConnector())
```

That's it - the instance automatically gets: `/v1/models` listing, UI
provider chips, routing-chain eligibility, rate limiting, auth, metrics
labels, and (when tracing is enabled) per-call OTel spans.

## 5. Testing recipe (mock-only - never live keys)

Follow `tests/test_connectors.py`:

- Stub the SDK boundary (monkeypatch the client object) - no network.
- Cover: success shape, timeout passthrough, quota -> RATE_LIMITED,
  missing-key ERROR response.
- API-level: register a stub via `monkeypatch.setattr(registry,
  "_connectors", {...})` inside `with TestClient(app)` and assert
  envelope behavior.

The scaffold script generates this test file for you.

## 6. Checklist before opening the PR

- [ ] `ruff check .` and `mypy app scripts` clean
- [ ] Generated tests pass; added provider-specific cases
- [ ] `config.py` + `.env.template` entries added
- [ ] Registered in `main.py`
- [ ] README provider table updated
- [ ] No secrets committed - keys live only in `.env`
