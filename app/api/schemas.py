from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ConnectorConfigRequest(BaseModel):
    connectors: list[str] | None = Field(
        default=None,
        description="Connector IDs to activate. Omit to use every available connector.",
    )
    profile: str | None = Field(
        default=None,
        description=(
            "Named routing profile from config/routing.yaml (e.g. 'research', "
            "'code'). Ignored when connectors is set."
        ),
    )
    role_bindings: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Per-request role-to-provider preference overrides. Keys: "
            "researcher, analyzer, verifier, synthesizer. Values are ordered "
            "connector IDs."
        ),
    )
    router_strategy: str | None = Field(
        default=None,
        description=(
            "Routing strategy override: 'static' (fixed YAML chains) or "
            "'semantic' (infer a profile from the query). Defaults to the "
            "ROUTER_STRATEGY setting."
        ),
    )
    timeout_s: int = Field(default=45, ge=5, le=300)
    max_tokens: int = Field(default=4096, ge=256, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=32000)
    session_id: str | None = Field(default=None, max_length=128)
    model_config_: ConnectorConfigRequest = Field(
        default_factory=ConnectorConfigRequest,
        alias="model_config",
    )

    model_config = {"populate_by_name": True}


class TokenUsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelStatus(BaseModel):
    role: Literal["researcher", "analyzer", "verifier", "direct"]
    connector_id: str
    status: Literal["success", "timeout", "error", "rate_limited", "skipped"]
    latency_ms: int
    error: str | None = None
    token_usage: TokenUsageOut | None = None
    sub_query: str | None = None
    retry_after_s: float | None = None


class QueryResponse(BaseModel):
    request_id: str
    query: str
    result: str
    synthesizer: str
    model_statuses: list[ModelStatus]
    latency_breakdown: dict[str, int]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    short_circuited: bool = False
    role_assignments: dict[str, str] = Field(default_factory=dict)
    cache_hit: bool = False
    router_strategy: str | None = None
    matched_profile: str | None = None
    session_id: str | None = None


class ConnectorProfile(BaseModel):
    connector_id: str
    display_name: str
    capabilities: list[str]
    is_available: bool
    demoted: bool = False
    consecutive_auth_failures: int = 0


class ModelsResponse(BaseModel):
    connectors: list[ConnectorProfile]
    total: int


class RoutingStrategyOut(BaseModel):
    name: str
    description: str


class RoutingProfileOut(BaseModel):
    name: str
    connectors: list[str]
    description: str


class RoutingInfoResponse(BaseModel):
    strategies: list[RoutingStrategyOut]
    profiles: list[RoutingProfileOut]


class ConnectorHealthStatus(BaseModel):
    connector_id: str
    is_available: bool
    status: Literal["ok", "degraded", "unavailable"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    connectors: list[ConnectorHealthStatus]
    redis: Literal["ok", "unavailable", "disabled"] = "disabled"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionTurn(BaseModel):
    q: str
    a: str
    ts: float


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=128)
    rating: int = Field(..., ge=1, le=5)


class FeedbackResponse(BaseModel):
    request_id: str
    rating: int
    stored: bool


class SessionDetail(BaseModel):
    session_id: str
    turns: list[SessionTurn]
    enabled: bool


class TranscriptionResponse(BaseModel):
    text: str
    language_code: str | None = None
    model: str
    latency_ms: int


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AudioQueryResponse(QueryResponse):
    transcript_text: str
    transcript_language_code: str | None = None
    transcript_model: str
    transcription_latency_ms: int


class ReportCreateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running"] = "queued"
    poll_url: str


class ReportJobStatus(BaseModel):
    job_id: str
    query: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: str
    result_markdown: str | None = None
    error: str | None = None
    role_assignments: dict[str, str] = Field(default_factory=dict)
    created_at: float
    updated_at: float


# P4-0 investigate API schemas (append-only section; existing classes above are untouched).
from app.analysis.synthesis import SynthesisRecord  # noqa: E402
from app.evidence.models import (  # noqa: E402
    Claim,
    Evidence,
    InvestigationStatus,
    StatusReason,
)


class InvestigateRequest(BaseModel):
    # "local" is an identity placeholder and MUST NOT be treated as proof of authorization.
    query: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(default="local", min_length=1, max_length=128)


class InvestigateCreated(BaseModel):
    investigation_id: str
    user_id: str
    status: InvestigationStatus
    status_reason: StatusReason | None = None


class BoardCounts(BaseModel):
    evidence: int
    claims: int


class InvestigationBoardResponse(BaseModel):
    investigation_id: str
    user_id: str
    query: str
    status: InvestigationStatus
    status_reason: StatusReason | None = None
    created_at: float
    updated_at: float
    schema_version: str
    evidence: list[Evidence]
    claims: list[Claim]
    counts: BoardCounts
    truncated: bool
    # P4-3 additive field (locked decision): milestone syntheses, oldest first.
    syntheses: list[SynthesisRecord] = Field(default_factory=list)


class CancelInvestigationResponse(BaseModel):
    investigation_id: str
    user_id: str
    status: InvestigationStatus
    status_reason: StatusReason | None = None


class InvestigationSummary(BaseModel):
    investigation_id: str
    user_id: str
    query: str
    status: InvestigationStatus
    status_reason: StatusReason | None = None
    created_at: float
    updated_at: float
    evidence_count: int
    claim_count: int
    synthesis_count: int


class InvestigationListResponse(BaseModel):
    investigations: list[InvestigationSummary]


class InvestigateFeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)


class InvestigateFeedbackResponse(BaseModel):
    investigation_id: str
    rating: int
    stored: bool
