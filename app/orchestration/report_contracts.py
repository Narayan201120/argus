"""Typed contracts for the deep-report pipeline."""

from pydantic import BaseModel, Field

from app.orchestration.contracts import (
    AnalysisResult,
    ResearchResult,
    SharedTaskState,
)

MAX_REPORT_SUBTASKS = 5


class ReportSubtask(BaseModel):
    subtask_id: str
    title: str
    objective: str
    focus: list[str] = Field(default_factory=list)


class ReportPlan(BaseModel):
    report_title: str
    shared_state: SharedTaskState
    subtasks: list[ReportSubtask]


class TrackResult(BaseModel):
    """Research + analysis outputs for one planned subtask."""

    subtask_id: str
    title: str
    research: ResearchResult | None = None
    analysis: AnalysisResult | None = None
    error: str | None = None


class ReviewVerdict(BaseModel):
    approved: bool = False
    issues: list[str] = Field(default_factory=list)


class VerificationSummary(BaseModel):
    """Global verifier output for the whole report."""

    critical_risks: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
