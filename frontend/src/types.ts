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
