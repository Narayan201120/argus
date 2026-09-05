import { useEffect, useState } from 'react'
import { fetchPaper, fetchSimilar, searchPapers } from './radar'
import type {
  RadarPage,
  RadarPaperDetail,
  RadarPaperItem,
  RadarSimilarPaper,
} from './types'

const BOOKMARKS_KEY = 'argus.bookmarks'
const PAGE_SIZE = 10

function loadBookmarks(): string[] {
  try {
    const raw = localStorage.getItem(BOOKMARKS_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : []
  } catch {
    return []
  }
}

function statusOf(err: unknown): number | undefined {
  return (err as { status?: number } | null)?.status
}

function radarMessage(err: unknown): string {
  const status = statusOf(err)
  if (status === 404) return 'Radar workspace is disabled'
  if (status !== undefined && status >= 500) return 'Radar is unreachable'
  return err instanceof Error ? err.message : 'Radar request failed'
}

export default function RadarView() {
  const [tab, setTab] = useState<'search' | 'saved'>('search')
  const [q, setQ] = useState('')
  const [year, setYear] = useState('')
  const [topic, setTopic] = useState('')
  const [author, setAuthor] = useState('')
  const [page, setPage] = useState(1)
  const [data, setData] = useState<RadarPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<RadarPaperDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [similar, setSimilar] = useState<RadarSimilarPaper[]>([])
  const [bookmarks, setBookmarks] = useState<string[]>(loadBookmarks)
  const [saved, setSaved] = useState<RadarPage | null>(null)
  const [savedError, setSavedError] = useState<string | null>(null)

  async function runSearch(nextPage: number) {
    setLoading(true)
    setError(null)
    try {
      const result = await searchPapers({
        q: q.trim() || undefined,
        year: year.trim() || undefined,
        topic: topic.trim() || undefined,
        author: author.trim() || undefined,
        page: nextPage,
        page_size: PAGE_SIZE,
      })
      setPage(nextPage)
      setData(result)
    } catch (err) {
      setError(radarMessage(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // Initial load only; later searches run from the form submit handler.
    searchPapers({ page: 1, page_size: PAGE_SIZE })
      .then((result) => {
        setPage(1)
        setData(result)
      })
      .catch((err: unknown) => {
        setError(radarMessage(err))
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (!selectedId) return
    let cancelled = false
    const id = selectedId
    fetchPaper(id)
      .then((d) => {
        if (!cancelled) setDetail(d)
      })
      .catch((err: unknown) => {
        if (!cancelled) setDetailError(radarMessage(err))
      })
    fetchSimilar(id)
      .then((s) => {
        if (!cancelled) setSimilar(s)
      })
      .catch(() => {
        if (!cancelled) setSimilar([])
      })
    return () => {
      cancelled = true
    }
  }, [selectedId])

  async function loadSaved() {
    const ids = loadBookmarks()
    setBookmarks(ids)
    if (ids.length === 0) {
      setSaved(null)
      setSavedError(null)
      return
    }
    setSavedError(null)
    try {
      setSaved(await searchPapers({ ids: ids.join(','), page: 1, page_size: ids.length }))
    } catch (err) {
      setSavedError(radarMessage(err))
    }
  }

  function openSaved() {
    setTab('saved')
    loadSaved()
  }

  function selectPaper(id: string) {
    setSelectedId(id)
    setDetail(null)
    setDetailError(null)
    setSimilar([])
  }

  function toggleBookmark(id: string) {
    setBookmarks((prev) => {
      const next = prev.includes(id) ? prev.filter((b) => b !== id) : [...prev, id]
      try {
        localStorage.setItem(BOOKMARKS_KEY, JSON.stringify(next))
      } catch {
        /* persistence is best-effort */
      }
      return next
    })
  }

  function renderRow(p: RadarPaperItem) {
    const marked = bookmarks.includes(p.id)
    return (
      <div key={p.id} className="paper-card" onClick={() => selectPaper(p.id)}>
        <span className="title">{p.title}</span>
        {p.publication_year !== null && <span className="year">{p.publication_year}</span>}
        <span className="cites">{p.cited_by_count} cites</span>
        <span className="authors">{p.authors.map((a) => a.name).join(', ')}</span>
        <button
          className={`bookmark ${marked ? 'on' : ''}`}
          title={marked ? 'Remove bookmark' : 'Bookmark this paper'}
          onClick={(e) => {
            e.stopPropagation()
            toggleBookmark(p.id)
          }}
        >
          {marked ? '★' : '☆'}
        </button>
      </div>
    )
  }

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1

  return (
    <div className="view">
      <div className="radar-bar">
        <button className={`chip ${tab === 'search' ? 'active' : ''}`} onClick={() => setTab('search')}>
          Search
        </button>
        <button
          className={`chip ${tab === 'saved' ? 'active' : ''}`}
          onClick={openSaved}
          title="Bookmarked papers"
        >
          Saved ({bookmarks.length})
        </button>
      </div>

      {tab === 'search' ? (
        <>
          <form
            className="radar-bar"
            onSubmit={(e) => {
              e.preventDefault()
              runSearch(1)
            }}
          >
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search papers…" />
            <input
              className="filter"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              placeholder="Year"
            />
            <input
              className="filter"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Topic"
            />
            <input
              className="filter"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Author"
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </form>
          {error && <div className="failure">{error}</div>}
          {!error && data && data.items.length === 0 && !loading && (
            <div className="waiting">No papers found.</div>
          )}
          {data?.items.map(renderRow)}
          {data && data.total > 0 && (
            <div className="pager">
              <button disabled={loading || page <= 1} onClick={() => runSearch(page - 1)}>
                ← Prev
              </button>
              <span>
                page {page} of {totalPages} · {data.total} papers
              </span>
              <button disabled={loading || page >= totalPages} onClick={() => runSearch(page + 1)}>
                Next →
              </button>
            </div>
          )}
        </>
      ) : (
        <>
          {savedError && <div className="failure">{savedError}</div>}
          {!savedError && bookmarks.length === 0 && (
            <div className="waiting">No bookmarks yet. Star a paper to keep it here.</div>
          )}
          {saved?.items.map(renderRow)}
        </>
      )}

      {selectedId && (
        <div className="radar-detail">
          <div className="radar-bar">
            <button onClick={() => setSelectedId(null)}>Close detail</button>
            <button
              className={`bookmark ${bookmarks.includes(selectedId) ? 'on' : ''}`}
              onClick={() => toggleBookmark(selectedId)}
              title="Toggle bookmark"
            >
              {bookmarks.includes(selectedId) ? '★ saved' : '☆ save'}
            </button>
          </div>
          {detailError && <div className="failure">{detailError}</div>}
          {!detail && !detailError && <div className="waiting">Loading paper…</div>}
          {detail && (
            <>
              <div>
                <strong>{detail.title}</strong>
              </div>
              <div className="sub">
                {detail.publication_year ?? '—'} · {detail.cited_by_count} cites ·{' '}
                {detail.authors.map((a) => a.name).join(', ')}
              </div>
              {detail.abstract && <p>{detail.abstract}</p>}
              {detail.topics.length > 0 && (
                <div className="topics">
                  {detail.topics.map((t) => (
                    <span key={t} className="fact">
                      {t}
                    </span>
                  ))}
                </div>
              )}
              {detail.doi && (
                <a href={`https://doi.org/${detail.doi}`} target="_blank" rel="noreferrer">
                  {detail.doi}
                </a>
              )}
              <h4>Similar papers</h4>
              {similar.length === 0 && <div className="waiting">No similar papers.</div>}
              {similar.map((s) => (
                <div key={s.id} className="sim-row">
                  <span>{s.title}</span>
                  <span className="score">{s.similarity_score.toFixed(2)}</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}
