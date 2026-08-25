import type { ModelsResponse, QueryEnvelope, QueryOptions, RoutingInfo } from './types'

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

export async function getModels(): Promise<ModelsResponse> {
  return jsonOrThrow(await fetch('/v1/models'))
}

export async function getRouting(): Promise<RoutingInfo> {
  return jsonOrThrow(await fetch('/v1/routing'))
}

export async function transcribeAudio(file: File): Promise<string> {
  const form = new FormData()
  form.append('file', file)
  const body = await jsonOrThrow<{ text: string }>(
    await fetch('/v1/transcribe', { method: 'POST', body: form }),
  )
  return body.text
}

export async function speakText(text: string): Promise<string> {
  const response = await fetch('/v1/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
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
  return URL.createObjectURL(await response.blob())
}

/**
 * POST /v1/query/stream and parse the SSE wire format incrementally.
 * Events: role_complete, synthesis_start, synthesis_token, synthesis_end,
 * synthesis_fallback_concat, final, stream_error.
 */
export async function streamQuery(
  options: QueryOptions,
  handlers: {
    onRoleComplete?: (event: { role: string; connector_id: string; status: string; latency_ms: number; error: string | null }) => void
    onSynthesisStart?: (connectorId: string) => void
    onSynthesisToken?: (delta: string) => void
    onFallbackConcat?: () => void
    onFinal: (envelope: QueryEnvelope) => void
    onError: (message: string) => void
  },
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/v1/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: options.query,
      ...(options.sessionId ? { session_id: options.sessionId } : {}),
      model_config: {
        ...(options.connectors?.length ? { connectors: options.connectors } : {}),
        ...(options.profile ? { profile: options.profile } : {}),
        ...(options.router_strategy ? { router_strategy: options.router_strategy } : {}),
        ...(options.timeout_s ? { timeout_s: options.timeout_s } : {}),
      },
    }),
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

    switch (event) {
      case 'role_complete':
        handlers.onRoleComplete?.(payload as Parameters<NonNullable<typeof handlers.onRoleComplete>>[0])
        break
      case 'synthesis_start':
        handlers.onSynthesisStart?.((payload as { connector_id: string }).connector_id)
        break
      case 'synthesis_token':
        handlers.onSynthesisToken?.((payload as { delta: string }).delta)
        break
      case 'synthesis_end':
        break
      case 'synthesis_fallback_concat':
        handlers.onFallbackConcat?.()
        break
      case 'final':
        handlers.onFinal(payload as QueryEnvelope)
        break
      case 'stream_error':
        handlers.onError((payload as { error: string }).error)
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
