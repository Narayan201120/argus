import { useEffect, useState } from 'react'
import { listInvestigations } from './radar'
import type { InvestigationSummary } from './types'

interface Props {
  onOpenInvestigation: (id: string) => void
}

export default function LibraryView({ onOpenInvestigation }: Props) {
  const [items, setItems] = useState<InvestigationSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listInvestigations(20)
      .then((res) => {
        if (cancelled) return
        setItems([...res.investigations].sort((a, b) => b.created_at - a.created_at))
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'History unavailable')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="view">
      <div className="bubble">Document library arrives with RAG integration.</div>
      <h3 className="lib-head">Investigation history</h3>
      {loading && <div className="waiting">Loading history…</div>}
      {error && <div className="failure">{error}</div>}
      {!loading && !error && items.length === 0 && (
        <div className="waiting">No investigations yet.</div>
      )}
      {items.map((inv) => (
        <div
          key={inv.investigation_id}
          className="lib-item"
          onClick={() => onOpenInvestigation(inv.investigation_id)}
        >
          <div>{inv.query}</div>
          <div className="sub">
            {inv.status} · {inv.evidence_count} evidence · {inv.claim_count} claims ·{' '}
            {new Date(inv.created_at * 1000).toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  )
}
