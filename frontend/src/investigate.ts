import type {
  InvestigationBoard,
  InvestigationCancelResult,
  InvestigationClaim,
  InvestigationCreated,
  InvestigationEvidence,
  InvestigationStreamHandlers,
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
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export async function startInvestigation(query: string): Promise<InvestigationCreated> {
  return jsonOrThrow(
    await fetch('/v1/investigate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    }),
  )
}

export async function fetchBoard(investigationId: string, signal?: AbortSignal): Promise<InvestigationBoard> {
  return jsonOrThrow(await fetch(`/v1/investigate/${investigationId}`, { signal }))
}

export async function cancelInvestigation(investigationId: string): Promise<InvestigationCancelResult> {
  return jsonOrThrow(
    await fetch(`/v1/investigate/${investigationId}/cancel`, { method: 'POST' }),
  )
}

/**
 * GET /v1/investigate/{id}/stream and parse the SSE wire format incrementally.
 * Events: board_snapshot, round_started, evidence_added, claims_updated,
 * synthesis_start, synthesis_token, synthesis_end, terminal, stream_error.
 */
export async function streamInvestigation(
  investigationId: string,
  handlers: InvestigationStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/v1/investigate/${investigationId}/stream`, {
    headers: { Accept: 'text/event-stream' },
    signal,
  })

  if (!response.ok || !response.body) {
    await jsonOrThrow(response)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const dispatch = (block: string) => {
    let event = 'message'
    let data = ''
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) data += line.slice(5).trim()
    }
    if (!data) return

    let payload: unknown
    try {
      payload = JSON.parse(data)
    } catch {
      return
    }
    const body =
      payload !== null && typeof payload === 'object'
        ? (payload as Record<string, unknown>)
        : {}

    switch (event) {
      case 'board_snapshot':
        handlers.onBoardSnapshot?.(payload as InvestigationBoard)
        break
      case 'round_started':
        handlers.onRoundStarted?.(Number(body.round ?? body.iteration ?? 0))
        break
      case 'evidence_added':
        handlers.onEvidenceAdded?.(
          (body.evidence as InvestigationEvidence) ?? (payload as InvestigationEvidence),
        )
        break
      case 'claims_updated': {
        const claims = body.claims ?? (Array.isArray(payload) ? payload : [payload])
        handlers.onClaimsUpdated?.(claims as InvestigationClaim[])
        break
      }
      case 'synthesis_start':
        handlers.onSynthesisStart?.(String(body.milestone ?? ''))
        break
      case 'synthesis_token':
        handlers.onSynthesisToken?.(
          typeof body.delta === 'string'
            ? body.delta
            : typeof body.token === 'string'
              ? body.token
              : typeof payload === 'string'
                ? payload
                : '',
        )
        break
      case 'synthesis_end':
        handlers.onSynthesisEnd?.(String(body.milestone ?? ''))
        break
      case 'terminal':
        handlers.onTerminal?.({
          status: String(body.status ?? ''),
          reason: typeof body.reason === 'string' ? body.reason : null,
        })
        break
      case 'stream_error':
        handlers.onError?.(typeof body.error === 'string' ? body.error : 'stream failed')
        break
    }
  }

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let separator = buffer.indexOf('\n\n')
    while (separator !== -1) {
      dispatch(buffer.slice(0, separator))
      buffer = buffer.slice(separator + 2)
      separator = buffer.indexOf('\n\n')
    }
  }
  if (buffer.trim()) dispatch(buffer)
}
