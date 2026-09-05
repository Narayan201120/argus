import type {
  LibraryCollectionDetail,
  LibraryCollectionListResponse,
  LibraryDocumentListResponse,
  LibraryDocumentPreview,
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
    // Attach the HTTP status so callers can tell "disabled" (404) from "down" (5xx).
    const err = new Error(detail) as Error & { status: number }
    err.status = response.status
    throw err
  }
  return response.json() as Promise<T>
}

let libraryEnabled: boolean | null = null

async function ensureLibrary(): Promise<void> {
  if (libraryEnabled === null) {
    try {
      const meta = (await (await fetch('/v1/meta')).json()) as {
        workspace?: { rag?: boolean }
      }
      libraryEnabled = meta.workspace?.rag ?? true
    } catch {
      libraryEnabled = true
    }
  }
  if (!libraryEnabled) {
    const err = new Error('Document library is disabled.') as Error & { status: number }
    err.status = 404
    throw err
  }
}

export async function listDocuments(): Promise<LibraryDocumentListResponse> {
  await ensureLibrary()
  return jsonOrThrow(await fetch('/v1/library/documents'))
}

export async function fetchDocumentPreview(filename: string): Promise<LibraryDocumentPreview> {
  await ensureLibrary()
  return jsonOrThrow(await fetch(`/v1/library/documents/${encodeURIComponent(filename)}`))
}

export async function listCollections(): Promise<LibraryCollectionListResponse> {
  await ensureLibrary()
  return jsonOrThrow(await fetch('/v1/library/collections'))
}

export async function fetchCollection(id: string): Promise<LibraryCollectionDetail> {
  await ensureLibrary()
  return jsonOrThrow(await fetch(`/v1/library/collections/${encodeURIComponent(id)}`))
}
