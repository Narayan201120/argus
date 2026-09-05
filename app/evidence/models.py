"""P4-0 Evidence Board models (DEC-053, backend only, mock-only)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class InvestigationStatus(str, Enum):  # noqa: UP042 - contract mandates (str, Enum)
    PLANNED = "planned"
    GATHERING = "gathering"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class StatusReason(str, Enum):  # noqa: UP042 - contract mandates (str, Enum)
    ITERATION_LIMIT = "iteration_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    WALL_CLOCK_LIMIT = "wall_clock_limit"
    PROVIDER_FAILURE = "provider_failure"
    CANCELLED = "cancelled"
    SUFFICIENT_EVIDENCE = "sufficient_evidence"


class ClaimStatus(str, Enum):  # noqa: UP042 - contract mandates (str, Enum)
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    REJECTED = "rejected"


class Evidence(BaseModel):
    id: str = Field(min_length=1)
    investigation_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1, max_length=20000)
    type: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: float


class Claim(BaseModel):
    id: str = Field(min_length=1)
    investigation_id: str = Field(min_length=1)
    statement: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.PROPOSED


class Board(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


class BudgetLimits(BaseModel):
    max_iterations: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_wall_time_s: int = Field(gt=0)


class BudgetUsage(BaseModel):
    iterations_used: int = Field(default=0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    web_calls_used: int = Field(default=0, ge=0)


class Investigation(BaseModel):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    status: InvestigationStatus
    status_reason: StatusReason | None = None
    created_at: float
    updated_at: float
    deadline_at: float
    schema_version: str = SCHEMA_VERSION
    budgets: BudgetLimits
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    board: Board = Field(default_factory=Board)
