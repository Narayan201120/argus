import { useEffect, useRef, useState } from 'react'
import { getModels, getRouting, postFeedback, speakText, streamQuery, transcribeAudio } from './api'
import { startInvestigation } from './investigate'
import InvestigationView from './InvestigationView'
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
const SPEAK_CHAR_CAP = 1500 // mirrors SPEECH_MAX_CHARS default; server enforces too

interface SavedControls {
  connectors?: string[]
  strategy?: string
  profile?: string
  voiceOut?: boolean
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
  const [lastQuery, setLastQuery] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string>(() => {
    const existing = sessionStorage.getItem('argus.session')
    if (existing) return existing
    const fresh =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : `s-${Date.now()}-${Math.random().toString(16).slice(2)}`
    sessionStorage.setItem('argus.session', fresh)
    return fresh
  })
  const [pendingLabel, setPendingLabel] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [voiceOut, setVoiceOut] = useState(SAVED_CONTROLS.voiceOut ?? false)
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [speakingKey, setSpeakingKey] = useState<string | null>(null)
  const [ratings, setRatings] = useState<Record<string, number>>({})
  const [mode, setMode] = useState<'chat' | 'investigate'>('chat')
  const [investigationId, setInvestigationId] = useState<string | null>(null)
  const [invQuery, setInvQuery] = useState('')
  const [invBusy, setInvBusy] = useState(false)
  const [invFailure, setInvFailure] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (!pendingLabel) return
    let seconds = 0
    const timer = setInterval(() => {
      seconds += 1
      setElapsed(seconds)
    }, 1000)
    return () => clearInterval(timer)
  }, [pendingLabel])

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
        JSON.stringify({
          connectors: selectedConnectors,
          strategy,
          profile,
          voiceOut,
        }),
      )
    } catch {
      /* private mode etc. - persistence is best-effort */
    }
  }, [selectedConnectors, strategy, profile, voiceOut])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const available = models.filter((m) => m.is_available)

  function toggleConnector(id: string) {
    setSelectedConnectors((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id],
    )
  }

  function toggleMic() {
    if (recording) {
      recorderRef.current?.stop()
      return
    }
    if (typeof MediaRecorder === 'undefined') {
      window.alert('This browser does not support audio recording.')
      return
    }
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        const recorder = new MediaRecorder(stream)
        chunksRef.current = []
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) chunksRef.current.push(event.data)
        }
        recorder.onstop = () => {
          stream.getTracks().forEach((track) => track.stop())
          setRecording(false)
          const type = recorder.mimeType || 'audio/webm'
          const extension = type.includes('mp3')
            ? 'mp3'
            : type.includes('ogg')
              ? 'ogg'
              : 'webm'
          const file = new File([new Blob(chunksRef.current, { type })], `input.${extension}`, { type })
          setTranscribing(true)
          transcribeAudio(file)
            .then((text) => setInput((prev) => (prev ? `${prev} ${text}`.trim() : text)))
            .catch((err: Error) => window.alert(`Transcription failed: ${err.message}`))
            .finally(() => setTranscribing(false))
        }
        recorder.start()
        recorderRef.current = recorder
        setRecording(true)
      })
      .catch((err: Error) => window.alert(`Microphone unavailable: ${err.message}`))
  }

  async function playAnswer(key: string, text: string) {
    if (speakingKey === key) {
      audioRef.current?.pause()
      setSpeakingKey(null)
      return
    }
    audioRef.current?.pause()
    setSpeakingKey(key)
    try {
      // Credit care: cap spoken length regardless of answer size.
      const url = await speakText(text.slice(0, SPEAK_CHAR_CAP))
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onended = () => setSpeakingKey(null)
      await audio.play()
    } catch (err) {
      setSpeakingKey(null)
      window.alert(`Speech failed: ${(err as Error).message}`)
    }
  }

  function isFailedAnswer(message: ChatMessage): boolean {
    if (message.failure) return true
    if (!message.envelope) return false
    const statuses = message.envelope.model_statuses
    return (
      !message.envelope.result.trim() ||
      (statuses.length > 0 && statuses.every((s) => s.status !== 'success'))
    )
  }

  function pickAlternative(message: ChatMessage): string | null {
    const failed = new Set((message.envelope?.model_statuses ?? []).map((s) => s.connector_id))
    return available.find((m) => !failed.has(m.connector_id))?.connector_id ?? null
  }

  async function submit(text: string, pinsOverride?: string[]) {
    if (!text || busy) return
    const pins = pinsOverride !== undefined ? pinsOverride : selectedConnectors
    setLastQuery(text)
    setElapsed(0) // fresh timer per attempt
    setPendingLabel(pins.length ? pins.join(', ') : 'auto')
    if (pinsOverride !== undefined) setSelectedConnectors(pins)
    setInput('')
    setBusy(true)
    setMessages((prev) => [
      ...prev,
      { kind: 'user', content: text },
      { kind: 'assistant', content: '', streaming: true, trace: [] },
    ])

    abortRef.current = new AbortController()
    await streamQuery(
      {
        query: text,
        sessionId,
        connectors: pins.length ? pins : undefined,
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
          setPendingLabel(null) // tokens flowing - the wait is over
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
    }).finally(() => {
      setPendingLabel(null)
      setBusy(false)
      abortRef.current = null
    })
  }

  async function submitInvestigation(text: string) {
    if (!text || invBusy) return
    setInvBusy(true)
    setInvFailure(null)
    try {
      const created = await startInvestigation(text)
      setInvQuery('')
      setInvestigationId(created.investigation_id)
    } catch (err) {
      setInvFailure((err as Error).message)
    } finally {
      setInvBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>ARGUS</h1>
        <div className="controls">
          <button
            onClick={() => {
              const fresh =
                typeof crypto !== 'undefined' && 'randomUUID' in crypto
                  ? crypto.randomUUID()
                  : `s-${Date.now()}`
              sessionStorage.setItem('argus.session', fresh)
              setSessionId(fresh)
              setMessages([])
              setLastQuery(null)
            }}
            title="Start a fresh conversation (new session id)"
          >
            New chat
          </button>
          <button
            className={`chip ${mode === 'chat' ? 'active' : ''}`}
            onClick={() => setMode('chat')}
            title="Ask mode (existing chat flow)"
          >
            Chat
          </button>
          <button
            className={`chip ${mode === 'investigate' ? 'active' : ''}`}
            onClick={() => setMode('investigate')}
            title="Deep-research investigation mode"
          >
            Investigate
          </button>
          <button
            className={`chip ${voiceOut ? 'active' : ''}`}
            onClick={() => setVoiceOut((v) => !v)}
            title="Speak answers aloud (uses Sarvam TTS credits)"
          >
            🔊 voice {voiceOut ? 'on' : 'off'}
          </button>
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

      {mode === 'investigate' ? (
        <main className="thread">
          {investigationId ? (
            <InvestigationView investigationId={investigationId} onClose={() => setInvestigationId(null)} />
          ) : (
            <div className="bubble inv-start">
              <div className="waiting">Deep research: start an investigation, then watch evidence, claims and the report arrive live.</div>
              <textarea
                value={invQuery}
                placeholder="What should ARGUS investigate?"
                onChange={(e) => setInvQuery(e.target.value)}
                rows={3}
                disabled={invBusy}
              />
              {invFailure && <div className="failure">{invFailure}</div>}
              <div className="recover">
                <button onClick={() => submitInvestigation(invQuery.trim())} disabled={invBusy || !invQuery.trim()}>
                  {invBusy ? 'Starting…' : 'Start investigation'}
                </button>
              </div>
            </div>
          )}
        </main>
      ) : (
      <>
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
              {message.kind === 'assistant' &&
                message.streaming &&
                !message.content &&
                pendingLabel && (
                  <div className="waiting">
                    asking {pendingLabel}… {elapsed}s
                  </div>
                )}
            </div>

            {message.kind === 'assistant' &&
              !busy &&
              lastQuery &&
              isFailedAnswer(message) &&
              !message.streaming && (
                <div className="recover">
                  {pickAlternative(message) && (
                    <button onClick={() => submit(lastQuery, [pickAlternative(message)!])}>
                      Retry with {pickAlternative(message)}
                    </button>
                  )}
                  <button onClick={() => submit(lastQuery, [])}>Unpin &amp; retry all</button>
                </div>
              )}

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

            {message.kind === 'assistant' && message.envelope && !isFailedAnswer(message) && (
              <div className="meta rate">
                <span className="hint">rate:</span>
                {[1, 2, 3, 4, 5].map((value) => {
                  const id = message.envelope!.request_id
                  const selected = ratings[id] ?? 0
                  return (
                    <button
                      key={value}
                      className={`star ${selected >= value ? 'on' : ''}`}
                      onClick={() => {
                        postFeedback(id, value)
                          .then(() => setRatings((prev) => ({ ...prev, [id]: value })))
                          .catch(() => {})
                      }}
                      title={`Rate ${value}/5`}
                    >
                      ★
                    </button>
                  )
                })}
              </div>
            )}

            {message.kind === 'assistant' &&
              voiceOut &&
              !message.streaming &&
              !isFailedAnswer(message) &&
              message.content && (
                <div className="meta">
                  <button
                    className={`chip ${speakingKey === String(index) ? 'active' : ''}`}
                    onClick={() => playAnswer(String(index), message.content)}
                  >
                    {speakingKey === String(index) ? '⏹ stop' : '▶ speak'}
                  </button>
                </div>
              )}
          </article>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer>
        <button
          className={`mic ${recording ? 'recording' : ''}`}
          onClick={toggleMic}
          disabled={busy || transcribing}
          title={recording ? 'Stop recording' : 'Record a question'}
        >
          {recording ? '⏹' : transcribing ? '…' : '🎙'}
        </button>
        <textarea
          value={input}
          placeholder={busy ? 'ARGUS is thinking…' : 'Ask anything. Enter to send.'}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit(input.trim())
            }
          }}
          rows={2}
          disabled={busy}
        />
        <button onClick={() => submit(input.trim())} disabled={busy || !input.trim()}>
          Send
        </button>
        {busy && (
          <button className="stop" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        )}
      </footer>
      </>
      )}
    </div>
  )
}
