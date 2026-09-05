import { useEffect, useRef, useState } from 'react'
import { cancelInvestigation, fetchBoard, streamInvestigation } from './investigate'
import type { InvestigationBoard, InvestigationSynthesis } from './types'

interface Props {
  investigationId: string
  onClose: () => void
  initialId?: string | null
}

function isTerminalStatus(status: string): boolean {
  return (
    status === 'complete' ||
    status === 'cancelled' ||
    status === 'failed' ||
    status === 'budget_exhausted'
  )
}

export default function InvestigationView({ investigationId, onClose, initialId }: Props) {
  // External-open path: a parent can open an investigation it did not start
  // (chat handoff, history re-open) via initialId. Remount per id (key on the
  // id at the call site) so each investigation starts with clean state while
  // running streams survive tab switches.
  const effectiveId = initialId || investigationId
  const [board, setBoard] = useState<InvestigationBoard | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [streamText, setStreamText] = useState('')
  const [synthesizing, setSynthesizing] = useState(false)
  const [frozen, setFrozen] = useState(false)
  const [round, setRound] = useState<number | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [cancelling, setCancelling] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const frozenRef = useRef(false)
  const streamTextRef = useRef('')
  const milestoneRef = useRef('')
  const status = board?.status

  useEffect(() => {
    if (frozen || (status && isTerminalStatus(status))) return
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000)
    return () => clearInterval(timer)
  }, [frozen, status])

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller
    let settled = false

    const live = () => !settled && !frozenRef.current

    const freeze = () => {
      frozenRef.current = true
      setFrozen(true)
      setSynthesizing(false)
    }

    const foldStreaming = (milestone: string) => {
      const text = streamTextRef.current
      if (!text) return
      const entry: InvestigationSynthesis = {
        milestone,
        markdown: text,
        final: false,
        created_at: Date.now() / 1000,
      }
      setBoard((prev) => (prev ? { ...prev, syntheses: [...(prev.syntheses ?? []), entry] } : prev))
      streamTextRef.current = ''
      setStreamText('')
    }

    fetchBoard(effectiveId, controller.signal)
      .then((initial) => {
        if (!live()) return
        setBoard(initial)
        if (isTerminalStatus(initial.status)) {
          freeze()
          return
        }
        return streamInvestigation(
          effectiveId,
          {
            onBoardSnapshot: (snapshot) => {
              if (!live()) return
              setBoard(snapshot)
              if (isTerminalStatus(snapshot.status)) freeze()
            },
            onRoundStarted: (n) => {
              if (live()) setRound(n)
            },
            onEvidenceAdded: (item) => {
              if (!live()) return
              setBoard((prev) =>
                prev && !prev.evidence.some((e) => e.id === item.id)
                  ? { ...prev, evidence: [...prev.evidence, item] }
                  : prev,
              )
            },
            onClaimsUpdated: (claims) => {
              if (!live()) return
              setBoard((prev) => {
                if (!prev) return prev
                const byId = new Map(prev.claims.map((c) => [c.id, c]))
                for (const c of claims) byId.set(c.id, c)
                return { ...prev, claims: [...byId.values()] }
              })
            },
            onSynthesisStart: (milestone) => {
              if (!live()) return
              milestoneRef.current = milestone
              streamTextRef.current = ''
              setStreamText('')
              setSynthesizing(true)
            },
            onSynthesisToken: (delta) => {
              if (!live() || !delta) return
              streamTextRef.current += delta
              setStreamText((prev) => prev + delta)
            },
            onSynthesisEnd: (milestone) => {
              if (!live()) return
              setSynthesizing(false)
              foldStreaming(milestone || milestoneRef.current)
            },
            onTerminal: (event) => {
              if (!live()) return
              if (streamTextRef.current) foldStreaming(milestoneRef.current)
              setBoard((prev) =>
                prev ? { ...prev, status: event.status, status_reason: event.reason } : prev,
              )
              freeze()
            },
            onError: (message) => {
              if (live()) setFailure(message)
            },
          },
          controller.signal,
        )
      })
      .catch((err: Error) => {
        if (err.name === 'AbortError' || !live()) return
        setFailure(err.message)
      })

    return () => {
      settled = true
      controller.abort()
    }
  }, [effectiveId])

  async function stopResearch() {
    if (cancelling || frozen) return
    setCancelling(true)
    try {
      const result = await cancelInvestigation(effectiveId)
      setBoard((prev) =>
        prev ? { ...prev, status: result.status, status_reason: result.status_reason } : prev,
      )
    } catch (err) {
      setFailure((err as Error).message)
    } finally {
      setCancelling(false)
    }
  }

  if (!board && !failure) return <div className="waiting">Loading investigation…</div>
  if (!board) {
    return (
      <div className="inv">
        <div className="failure">{failure}</div>
        <div className="inv-actions">
          <button onClick={onClose}>Back</button>
        </div>
      </div>
    )
  }

  const syntheses = board.syntheses ?? []
  const latest = syntheses.length ? syntheses[syntheses.length - 1] : null
  const report = `${latest?.markdown ?? ''}${streamText}`
  const terminal = isTerminalStatus(board.status) || frozen

  return (
    <div className="inv">
      <div className="inv-head">
        <span className="inv-query">{board.query}</span>
        <span className="chip active">{board.status}</span>
        {round !== null && <span className="fact">round {round}</span>}
        <span className="fact">{elapsed}s</span>
        <span className="hint" />
        {!terminal && (
          <button className="stop" onClick={stopResearch} disabled={cancelling}>
            {cancelling ? 'Stopping…' : 'STOP RESEARCH'}
          </button>
        )}
        <button
          onClick={() => {
            abortRef.current?.abort()
            onClose()
          }}
        >
          Close
        </button>
      </div>
      {failure && <div className="failure">{failure}</div>}
      <div className="inv-grid">
        <section className="inv-col">
          <h3>Evidence ({board.counts?.evidence ?? board.evidence.length})</h3>
          {board.evidence.length === 0 && <div className="waiting">No evidence yet…</div>}
          {board.evidence.map((e) => (
            <div key={e.id} className="inv-item">
              <div className="ref">
                {e.source_ref} · {e.type} · {Math.round(e.confidence * 100)}%
              </div>
              <div>{e.content}</div>
            </div>
          ))}
        </section>
        <section className="inv-col">
          <h3>Claims ({board.counts?.claims ?? board.claims.length})</h3>
          {board.claims.length === 0 && <div className="waiting">No claims yet…</div>}
          {board.claims.map((c) => (
            <div
              key={c.id}
              className={c.status.toLowerCase() === 'contested' ? 'inv-item contested' : 'inv-item'}
            >
              <div>{c.statement}</div>
              <div className="claim-status">
                {c.status} · {Math.round(c.confidence * 100)}%
              </div>
            </div>
          ))}
        </section>
        <section className="inv-col">
          <h3>Report{synthesizing ? ' (writing…)' : ''}</h3>
          <div className="inv-item inv-report">
            {report || <span className="waiting">No synthesis yet…</span>}
            {synthesizing && <span className="cursor" />}
          </div>
        </section>
      </div>
      <div className="inv-actions">
        {board.status_reason && <span className="fact">{board.status_reason}</span>}
        {board.truncated && <span className="fact">truncated</span>}
        {terminal && <span className="fact">frozen</span>}
      </div>
    </div>
  )
}
