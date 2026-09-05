import { useEffect, useState } from 'react'
import {
  fetchCollection,
  fetchDocumentPreview,
  listCollections,
  listDocuments,
} from './library'
import { listInvestigations } from './radar'
import type {
  InvestigationSummary,
  LibraryCollectionDetail,
  LibraryCollectionListResponse,
  LibraryCollectionSummary,
  LibraryDocumentItem,
  LibraryDocumentListResponse,
  LibraryDocumentPreview,
} from './types'

interface Props {
  onOpenInvestigation: (id: string) => void
}

type Tab = 'documents' | 'collections' | 'history'

function statusOf(err: unknown): number | undefined {
  return (err as { status?: number } | null)?.status
}

function libraryMessage(err: unknown): string {
  const status = statusOf(err)
  if (status === 404) return 'Document library is disabled.'
  if (status !== undefined && status >= 500) return 'Document library is unreachable.'
  return err instanceof Error ? err.message : 'Library request failed'
}

function isDisabled(err: unknown): boolean {
  return statusOf(err) === 404
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function LibraryView({ onOpenInvestigation }: Props) {
  const [tab, setTab] = useState<Tab>('documents')

  // Documents tab state.
  const [docs, setDocs] = useState<LibraryDocumentListResponse | null>(null)
  const [docsLoading, setDocsLoading] = useState(true)
  const [docsError, setDocsError] = useState<string | null>(null)
  const [docsDisabled, setDocsDisabled] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
  const [preview, setPreview] = useState<LibraryDocumentPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  // Collections tab state.
  const [cols, setCols] = useState<LibraryCollectionListResponse | null>(null)
  const [colsLoading, setColsLoading] = useState(true)
  const [colsError, setColsError] = useState<string | null>(null)
  const [colsDisabled, setColsDisabled] = useState(false)
  const [selectedCol, setSelectedCol] = useState<string | null>(null)
  const [colDetail, setColDetail] = useState<LibraryCollectionDetail | null>(null)
  const [colDetailLoading, setColDetailLoading] = useState(false)
  const [colDetailError, setColDetailError] = useState<string | null>(null)

  // History section (pre-existing; kept as-is).
  const [items, setItems] = useState<InvestigationSummary[]>([])
  const [histLoading, setHistLoading] = useState(true)
  const [histError, setHistError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listDocuments()
      .then((res) => {
        if (!cancelled) setDocs(res)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (isDisabled(err)) setDocsDisabled(true)
        else setDocsError(libraryMessage(err))
      })
      .finally(() => {
        if (!cancelled) setDocsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    listCollections()
      .then((res) => {
        if (!cancelled) setCols(res)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        if (isDisabled(err)) setColsDisabled(true)
        else setColsError(libraryMessage(err))
      })
      .finally(() => {
        if (!cancelled) setColsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    listInvestigations(20)
      .then((res) => {
        if (cancelled) return
        setItems([...res.investigations].sort((a, b) => b.created_at - a.created_at))
      })
      .catch((err: unknown) => {
        if (!cancelled) setHistError(err instanceof Error ? err.message : 'History unavailable')
      })
      .finally(() => {
        if (!cancelled) setHistLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedDoc) return
    let cancelled = false
    const name = selectedDoc
    fetchDocumentPreview(name)
      .then((d) => {
        if (!cancelled) setPreview(d)
      })
      .catch((err: unknown) => {
        if (!cancelled) setPreviewError(libraryMessage(err))
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedDoc])

  useEffect(() => {
    if (!selectedCol) return
    let cancelled = false
    const id = selectedCol
    fetchCollection(id)
      .then((d) => {
        if (!cancelled) setColDetail(d)
      })
      .catch((err: unknown) => {
        if (!cancelled) setColDetailError(libraryMessage(err))
      })
      .finally(() => {
        if (!cancelled) setColDetailLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedCol])

  function selectDoc(name: string) {
    setSelectedDoc(name)
    setPreview(null)
    setPreviewError(null)
    setPreviewLoading(true)
  }

  function selectCollection(id: string) {
    setSelectedCol(id)
    setColDetail(null)
    setColDetailError(null)
    setColDetailLoading(true)
  }

  function openDocFromCollection(name: string) {
    setTab('documents')
    selectDoc(name)
  }

  function renderDocRow(doc: LibraryDocumentItem) {
    return (
      <div
        key={doc.name}
        className={`paper-card${selectedDoc === doc.name ? ' selected' : ''}`}
        onClick={() => selectDoc(doc.name)}
      >
        <span className="title">{doc.name}</span>
        <span className="cites">{formatSize(doc.size_bytes)}</span>
      </div>
    )
  }

  function renderCollectionRow(col: LibraryCollectionSummary) {
    return (
      <div
        key={col.id}
        className={`paper-card${selectedCol === col.id ? ' selected' : ''}`}
        onClick={() => selectCollection(col.id)}
      >
        <span className="title">{col.name}</span>
        <span className="cites">
          {col.document_count} doc{col.document_count === 1 ? '' : 's'}
        </span>
      </div>
    )
  }

  return (
    <div className="view">
      <div className="radar-bar">
        <button
          className={`chip ${tab === 'documents' ? 'active' : ''}`}
          onClick={() => setTab('documents')}
        >
          Documents
        </button>
        <button
          className={`chip ${tab === 'collections' ? 'active' : ''}`}
          onClick={() => setTab('collections')}
        >
          Collections
        </button>
        <button
          className={`chip ${tab === 'history' ? 'active' : ''}`}
          onClick={() => setTab('history')}
        >
          History
        </button>
      </div>

      {tab === 'documents' && (
        <>
          {docsLoading && <div className="waiting">Loading documents…</div>}
          {!docsLoading && docsDisabled && (
            <div className="waiting">Document library is disabled.</div>
          )}
          {!docsLoading && docsError && <div className="failure">{docsError}</div>}
          {!docsLoading && !docsDisabled && !docsError && (docs?.documents.length ?? 0) === 0 && (
            <div className="waiting">No documents in the library.</div>
          )}
          {docs?.documents.map(renderDocRow)}
          {selectedDoc && (
            <div className="radar-detail">
              <div className="radar-bar">
                <button onClick={() => setSelectedDoc(null)}>Close preview</button>
              </div>
              {previewLoading && <div className="waiting">Loading preview…</div>}
              {previewError && <div className="failure">{previewError}</div>}
              {preview && (
                <>
                  <div>
                    <strong>{preview.name}</strong>
                  </div>
                  <div className="sub">
                    {preview.extension} · {preview.total_characters.toLocaleString()} characters
                  </div>
                  {preview.truncated && (
                    <div className="waiting">
                      Preview truncated — showing part of {preview.total_characters.toLocaleString()}{' '}
                      characters.
                    </div>
                  )}
                  <pre className="lib-preview">{preview.content}</pre>
                </>
              )}
            </div>
          )}
        </>
      )}

      {tab === 'collections' && (
        <>
          {colsLoading && <div className="waiting">Loading collections…</div>}
          {!colsLoading && colsDisabled && (
            <div className="waiting">Document library is disabled.</div>
          )}
          {!colsLoading && colsError && <div className="failure">{colsError}</div>}
          {!colsLoading && !colsDisabled && !colsError && (cols?.collections.length ?? 0) === 0 && (
            <div className="waiting">No collections yet.</div>
          )}
          {cols?.collections.map(renderCollectionRow)}
          {selectedCol && (
            <div className="radar-detail">
              <div className="radar-bar">
                <button onClick={() => setSelectedCol(null)}>Close collection</button>
              </div>
              {colDetailLoading && <div className="waiting">Loading collection…</div>}
              {colDetailError && <div className="failure">{colDetailError}</div>}
              {colDetail && (
                <>
                  <div>
                    <strong>{colDetail.name}</strong>
                  </div>
                  {colDetail.description && <div className="sub">{colDetail.description}</div>}
                  <h4>
                    {colDetail.documents.length} document
                    {colDetail.documents.length === 1 ? '' : 's'}
                  </h4>
                  {colDetail.documents.length === 0 && (
                    <div className="waiting">This collection has no documents.</div>
                  )}
                  {colDetail.documents.map((doc) => (
                    <div
                      key={doc.name}
                      className="paper-card"
                      onClick={() => openDocFromCollection(doc.name)}
                      title="Open preview"
                    >
                      <span className="title">{doc.name}</span>
                      <span className="cites">{formatSize(doc.size_bytes)}</span>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </>
      )}

      {tab === 'history' && (
        <>
          <h3 className="lib-head">Investigation history</h3>
          {histLoading && <div className="waiting">Loading history…</div>}
          {histError && <div className="failure">{histError}</div>}
          {!histLoading && !histError && items.length === 0 && (
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
        </>
      )}
    </div>
  )
}
