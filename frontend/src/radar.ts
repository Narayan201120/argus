import type {
  InvestigationListResponse,
  RadarPage,
  RadarPaperDetail,
  RadarSearchParams,
  RadarSimilarPaper,
} from './types'

async function jsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* keep status-code detail */
    }
    // Attach the HTTP status so callers can tell "disabled" (404) from "down" (502).
    const err = new Error(detail) as Error & { status: number }
    err.status = response.status
    throw err
  }
  return response.json() as Promise<T>
}

function toQuery(params: RadarSearchParams): string {
  const qs = new URLSearchParams()
  if (params.q) qs.set('q', params.q)
  if (params.year) qs.set('year', params.year)
  if (params.topic) qs.set('topic', params.topic)
  if (params.author) qs.set('author', params.author)
  if (params.ids) qs.set('ids', params.ids)
  if (params.page !== undefined) qs.set('page', String(params.page))
  if (params.page_size !== undefined) qs.set('page_size', String(params.page_size))
  const s = qs.toString()
  return s ? `?${s}` : ''
}

export async function searchPapers(params: RadarSearchParams = {}): Promise<RadarPage> {
  return jsonOrThrow(await fetch(`/v1/radar/papers${toQuery(params)}`))
}

export async function fetchPaper(id: string): Promise<RadarPaperDetail> {
  return jsonOrThrow(await fetch(`/v1/radar/papers/${encodeURIComponent(id)}`))
}

export async function fetchSimilar(id: string): Promise<RadarSimilarPaper[]> {
  return jsonOrThrow(await fetch(`/v1/radar/papers/${encodeURIComponent(id)}/similar`))
}

export async function listInvestigations(limit = 20): Promise<InvestigationListResponse> {
  return jsonOrThrow(await fetch(`/v1/investigations?limit=${limit}`))
}
