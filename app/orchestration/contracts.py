from typing import Literal

from pydantic import BaseModel, Field


ConfidenceLevel = Literal["low", "medium", "high"]


class SharedTaskState(BaseModel):
    request_id: str
    original_query: str
    main_objective: str
    task_context: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    global_rules: list[str] = Field(default_factory=list)
    expected_final_output: str


class BaseRoleTask(BaseModel):
    objective: str
    scope: list[str] = Field(default_factory=list)
    do_not_cover: list[str] = Field(default_factory=list)
    required_output_fields: list[str] = Field(default_factory=list)


class ResearchTask(BaseRoleTask):
    role: Literal["researcher"] = "researcher"


class AnalysisTask(BaseRoleTask):
    role: Literal["analyzer"] = "analyzer"


class VerificationTask(BaseRoleTask):
    role: Literal["verifier"] = "verifier"


class ResearchResult(BaseModel):
    facts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"


class AnalysisResult(BaseModel):
    proposed_solution: str
    assumptions: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    critical_risks: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"


class AggregationInput(BaseModel):
    shared_state: SharedTaskState
    research_result: ResearchResult
    analysis_result: AnalysisResult
    verification_result: VerificationResult


class OrchestrationPlan(BaseModel):
    shared_state: SharedTaskState
    research_task: ResearchTask
    analysis_task: AnalysisTask
    verification_task: VerificationTask
