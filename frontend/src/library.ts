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

export async function listDocuments(): Promise<LibraryDocumentListResponse> {
  return jsonOrThrow(await fetch('/v1/library/documents'))
}

export async function fetchDocumentPreview(filename: string): Promise<LibraryDocumentPreview> {
  return jsonOrThrow(await fetch(`/v1/library/documents/${encodeURIComponent(filename)}`))
}

export async function listCollections(): Promise<LibraryCollectionListResponse> {
  return jsonOrThrow(await fetch('/v1/library/collections'))
}

export async function fetchCollection(id: string): Promise<LibraryCollectionDetail> {
  return jsonOrThrow(await fetch(`/v1/library/collections/${encodeURIComponent(id)}`))
}
