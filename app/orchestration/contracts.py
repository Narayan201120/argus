from typing import Literal

from pydantic import BaseModel, Field, field_validator

ConfidenceLevel = Literal["low", "medium", "high"]


def _clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _clean_text_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _clean_text(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


class SharedTaskState(BaseModel):
    request_id: str
    original_query: str
    main_objective: str
    task_context: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    global_rules: list[str] = Field(default_factory=list)
    expected_final_output: str

    @field_validator("request_id", "original_query", "main_objective", "expected_final_output", mode="before")
    @classmethod
    def _normalize_text_fields(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("task_context", "constraints", "global_rules", mode="before")
    @classmethod
    def _normalize_text_list_fields(cls, value: list[str]) -> list[str]:
        return _clean_text_list(value or [])


class BaseRoleTask(BaseModel):
    objective: str
    scope: list[str] = Field(default_factory=list)
    do_not_cover: list[str] = Field(default_factory=list)
    required_output_fields: list[str] = Field(default_factory=list)

    @field_validator("objective", mode="before")
    @classmethod
    def _normalize_objective(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("scope", "do_not_cover", "required_output_fields", mode="before")
    @classmethod
    def _normalize_list_fields(cls, value: list[str]) -> list[str]:
        return _clean_text_list(value or [])


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

    @field_validator("facts", "constraints", "references", "unknowns", mode="before")
    @classmethod
    def _normalize_result_lists(cls, value: list[str]) -> list[str]:
        return _clean_text_list(value or [])


class AnalysisResult(BaseModel):
    proposed_solution: str
    assumptions: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    validation_checks: list[str] = Field(default_factory=list)

    @field_validator("proposed_solution", mode="before")
    @classmethod
    def _normalize_solution(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("assumptions", "tradeoffs", "risks", "validation_checks", mode="before")
    @classmethod
    def _normalize_analysis_lists(cls, value: list[str]) -> list[str]:
        return _clean_text_list(value or [])


class VerificationResult(BaseModel):
    critical_risks: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "medium"

    @field_validator(
        "critical_risks",
        "hidden_assumptions",
        "edge_cases",
        "validation_requirements",
        mode="before",
    )
    @classmethod
    def _normalize_verification_lists(cls, value: list[str]) -> list[str]:
        return _clean_text_list(value or [])


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
