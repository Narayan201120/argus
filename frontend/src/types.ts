export interface ModelInfo {
  connector_id: string
  display_name: string
  capabilities: string[]
  is_available: boolean
}

export interface ModelsResponse {
  connectors: ModelInfo[]
  total: number
}

export interface RoutingStrategy {
  name: string
  description: string
}

export interface RoutingProfile {
  name: string
  connectors: string[]
  description: string
}

export interface RoutingInfo {
  strategies: RoutingStrategy[]
  profiles: RoutingProfile[]
}

export type RoleName = 'researcher' | 'analyzer' | 'verifier' | 'direct'

export interface RoleStatus {
  role: RoleName
  connector_id: string
  status: 'success' | 'timeout' | 'error' | 'rate_limited' | 'skipped'
  latency_ms: number
  error: string | null
  retry_after_s?: number | null
  token_usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null
}

export interface QueryEnvelope {
  request_id: string
  query: string
  result: string
  synthesizer: string
  model_statuses: RoleStatus[]
  latency_breakdown: Record<string, number>
  short_circuited: boolean
  role_assignments: Record<string, string>
  cache_hit: boolean
  router_strategy: string | null
  matched_profile: string | null
}

export interface QueryOptions {
  query: string
  sessionId?: string
  connectors?: string[]
  profile?: string
  router_strategy?: string
  timeout_s?: number
}

// P4-3 investigation view types (append-only; existing types above are untouched).

export type InvestigationStatus =
  | 'planned'
  | 'gathering'
  | 'analyzing'
  | 'synthesizing'
  | 'complete'
  | 'cancelled'
  | 'failed'
  | 'budget_exhausted'

export interface InvestigationEvidence {
  id: string
  source_ref: string
  content: string
  type: string
  confidence: number
}

export interface InvestigationClaim {
  id: string
  statement: string
  confidence: number
  evidence_ids: string[]
  status: string
}

export interface InvestigationSynthesis {
  milestone: string
  markdown: string
  final: boolean
  created_at: number
}

export interface InvestigationCounts {
  evidence: number
  claims: number
}

export interface InvestigationBoard {
  investigation_id: string
  user_id: string
  query: string
  status: string
  status_reason: string | null
  created_at: number
  updated_at: number
  schema_version: string
  evidence: InvestigationEvidence[]
  claims: InvestigationClaim[]
  counts: InvestigationCounts
  truncated: boolean
  syntheses: InvestigationSynthesis[]
}

export interface InvestigationCreated {
  investigation_id: string
  user_id: string
  status: string
  status_reason?: string | null
}

export interface InvestigationCancelResult {
  investigation_id: string
  user_id: string
  status: string
  status_reason: string | null
}

export interface InvestigationTerminalEvent {
  status: string
  reason: string | null
}

export interface InvestigationStreamHandlers {
  onBoardSnapshot?: (board: InvestigationBoard) => void
  onRoundStarted?: (round: number) => void
  onEvidenceAdded?: (evidence: InvestigationEvidence) => void
  onClaimsUpdated?: (claims: InvestigationClaim[]) => void
  onSynthesisStart?: (milestone: string) => void
  onSynthesisToken?: (delta: string) => void
  onSynthesisEnd?: (milestone: string) => void
  onTerminal?: (event: InvestigationTerminalEvent) => void
  onError?: (message: string) => void
}

// P4-4 radar workspace + investigation history types (append-only; existing types above are untouched).

export interface RadarAuthor {
  id: string
  name: string
}

export interface RadarPaperItem {
  id: string
  title: string
  publication_year: number | null
  cited_by_count: number
  authors: RadarAuthor[]
}

export interface RadarPage {
  items: RadarPaperItem[]
  total: number
  page: number
  page_size: number
}

export interface RadarPaperDetail {
  id: string
  title: string
  abstract: string | null
  publication_year: number | null
  doi: string | null
  cited_by_count: number
  authors: RadarAuthor[]
  topics: string[]
}

export interface RadarSimilarPaper {
  id: string
  title: string
  similarity_score: number
}

export interface RadarSearchParams {
  q?: string
  year?: string
  topic?: string
  author?: string
  page?: number
  page_size?: number
  ids?: string
}

export interface InvestigationSummary {
  investigation_id: string
  user_id: string
  query: string
  status: string
  status_reason: string | null
  created_at: number
  updated_at: number
  evidence_count: number
  claim_count: number
  synthesis_count: number
}

export interface InvestigationListResponse {
  investigations: InvestigationSummary[]
}

// P4-4 RAG document library types (append-only; existing types above are untouched).

export interface LibraryDocumentItem {
  name: string
  size_bytes: number
}

export interface LibraryDocumentListResponse {
  count: number
  documents: LibraryDocumentItem[]
}

export interface LibraryDocumentPreview {
  name: string
  extension: string
  content: string
  total_characters: number
  truncated: boolean
}

export interface LibraryCollectionSummary {
  id: string
  name: string
  description: string
  document_count: number
  created_at: number
}

export interface LibraryCollectionListResponse {
  count: number
  collections: LibraryCollectionSummary[]
}

export interface LibraryCollectionDetail {
  id: string
  name: string
  description: string
  created_at: number
  documents: LibraryDocumentItem[]
}
