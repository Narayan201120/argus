import { useEffect, useRef, useState } from 'react'
import { getModels, getRouting, streamQuery } from './api'
import type {
  ModelInfo,
  QueryEnvelope,
  RoleStatus,
  RoutingInfo,
} from './types'

interface TraceEntry {
  role: string
  connectorId: string
  status: string
  latencyMs: number
  error?: string | null
}

interface ChatMessage {
  kind: 'user' | 'assistant'
  content: string
  streaming?: boolean
  trace?: TraceEntry[]
  envelope?: QueryEnvelope
  failure?: string
}

const STATUS_CLASS: Record<string, string> = {
  success: 'ok',
  rate_limited: 'warn',
  timeout: 'warn',
  error: 'bad',
  skipped: 'muted',
}

const CONTROLS_KEY = 'argus.controls'

interface SavedControls {
  connectors?: string[]
  strategy?: string
  profile?: string
}

function loadSavedControls(): SavedControls {
  try {
    const raw = localStorage.getItem(CONTROLS_KEY)
    return raw ? (JSON.parse(raw) as SavedControls) : {}
  } catch {
    return {}
  }
}

// Evaluated once at module load: initial control state from last session.
const SAVED_CONTROLS: SavedControls = loadSavedControls()

function totalTokens(statuses?: RoleStatus[]): number {
  if (!statuses) return 0
  return statuses.reduce((sum, s) => sum + (s.token_usage?.total_tokens ?? 0), 0)
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [routing, setRouting] = useState<RoutingInfo | null>(null)
  const [selectedConnectors, setSelectedConnectors] = useState<string[]>(SAVED_CONTROLS.connectors ?? [])
  const [strategy, setStrategy] = useState(SAVED_CONTROLS.strategy ?? '')
  const [profile, setProfile] = useState(SAVED_CONTROLS.profile ?? '')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    getModels().then((m) => setModels(m.connectors)).catch(() => {})
    getRouting().then((r) => {
      setRouting(r)
      // Respect a persisted strategy; only default when none saved.
      setStrategy((current) => current || r.strategies[0]?.name || '')
    }).catch(() => {})
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(
        CONTROLS_KEY,
        JSON.stringify({ connectors: selectedConnectors, strategy, profile }),
      )
    } catch {
      /* private mode etc. - persistence is best-effort */
    }
  }, [selectedConnectors, strategy, profile])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const available = models.filter((m) => m.is_available)

  function toggleConnector(id: string) {
    setSelectedConnectors((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id],
    )
  }

  async function send() {
    const query = input.trim()
    if (!query || busy) return
    setInput('')
    setBusy(true)
    setMessages((prev) => [
      ...prev,
      { kind: 'user', content: query },
      { kind: 'assistant', content: '', streaming: true, trace: [] },
    ])

    abortRef.current = new AbortController()
    await streamQuery(
      {
        query,
        connectors: selectedConnectors.length ? selectedConnectors : undefined,
        profile: profile || undefined,
        router_strategy: strategy,
        timeout_s: 120,
      },
      {
        onRoleComplete: (event) => {
          setMessages((prev) => {
            const copy = [...prev]
            const last = { ...copy[copy.length - 1] }
            last.trace = [
              ...(last.trace ?? []),
              {
                role: event.role,
                connectorId: event.connector_id,
                status: event.status,
                latencyMs: event.latency_ms,
                error: event.error,
              },
            ]
            copy[copy.length - 1] = last
            return copy
          })
        },
        onSynthesisToken: (delta) => {
          setMessages((prev) => {
            const copy = [...prev]
            const last = { ...copy[copy.length - 1] }
            last.content += delta
            copy[copy.length - 1] = last
            return copy
          })
        },
        onFallbackConcat: () => {},
        onFinal: (envelope: QueryEnvelope) => {
          setMessages((prev) => {
            const copy = [...prev]
            const last = { ...copy[copy.length - 1] }
            last.content = envelope.result || last.content
            last.envelope = envelope
            last.streaming = false
            copy[copy.length - 1] = last
            return copy
          })
        },
        onError: (message) => {
          setMessages((prev) => {
            const copy = [...prev]
            const last = { ...copy[copy.length - 1] }
            last.failure = message
            last.streaming = false
            copy[copy.length - 1] = last
            return copy
          })
        },
      },
      abortRef.current.signal,
    ).catch((err: Error) => {
      if (err.name === 'AbortError') return
      setMessages((prev) => {
        const copy = [...prev]
        const last = { ...copy[copy.length - 1] }
        last.failure = err.message
        last.streaming = false
        copy[copy.length - 1] = last
        return copy
      })
    })

    setBusy(false)
    abortRef.current = null
  }

  return (
    <div className="app">
      <header>
        <h1>ARGUS</h1>
        <div className="controls">
          <label>
            Strategy
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {(routing?.strategies ?? []).map((s) => (
                <option key={s.name} value={s.name} title={s.description}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Profile
            <select value={profile} onChange={(e) => setProfile(e.target.value)}>
              <option value="">(none)</option>
              {(routing?.profiles ?? []).map((p) => (
                <option key={p.name} value={p.name} title={p.description}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <div className="connector-bar">
        <span>Providers:</span>
        {available.map((m: ModelInfo) => (
          <button
            key={m.connector_id}
            className={`chip ${selectedConnectors.includes(m.connector_id) ? 'active' : ''}`}
            onClick={() => toggleConnector(m.connector_id)}
          >
            {m.connector_id}
          </button>
        ))}
        <span className="hint">
          {selectedConnectors.length === 0 ? 'all available' : `${selectedConnectors.length} pinned`}
        </span>
      </div>

      <main className="thread">
        {messages.length === 0 && (
          <div className="empty">Ask ARGUS anything. Complex questions run researcher/analyzer/verifier in parallel.</div>
        )}
        {messages.map((message, index) => (
          <article key={index} className={`message ${message.kind}`}>
            <div className="bubble">
              {message.kind === 'user'
                ? message.content
                : message.content || (message.streaming ? <span className="cursor" /> : '')}
              {message.failure && <div className="failure">{message.failure}</div>}
            </div>

            {message.kind === 'assistant' && (message.trace?.length || message.envelope) && (
              <div className="meta">
                {(message.trace ?? []).map((t, i) => (
                  <span key={i} className={`role-chip ${STATUS_CLASS[t.status] ?? ''}`}>
                    {t.role}:{t.connectorId} · {t.status} · {t.latencyMs}ms
                    {t.status === 'rate_limited' && t.error?.includes('retry_after_s=')
                      ? ` (retry in ${t.error.split('retry_after_s=')[1]?.split(')')[0]}s)`
                      : ''}
                  </span>
                ))}
                {message.envelope && (
                  <>
                    <span className="fact">
                      {message.envelope.cache_hit ? 'cache HIT' : `total ${message.envelope.latency_breakdown.total_ms}ms`}
                    </span>
                    <span className="fact">{totalTokens(message.envelope.model_statuses)} tokens</span>
                    <span className="fact">
                      synth: {message.envelope.synthesizer}
                      {message.envelope.matched_profile ? ` · ${message.envelope.router_strategy}/${message.envelope.matched_profile}` : ''}
                    </span>
                  </>
                )}
              </div>
            )}
          </article>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer>
        <textarea
          value={input}
          placeholder={busy ? 'ARGUS is thinking…' : 'Ask anything. Enter to send.'}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          rows={2}
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim()}>
          Send
        </button>
        {busy && (
          <button className="stop" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        )}
      </footer>
    </div>
  )
}
