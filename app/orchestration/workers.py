import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
)
from app.orchestration.contracts import (
    AnalysisResult,
    AnalysisTask,
    ResearchResult,
    ResearchTask,
    SharedTaskState,
    VerificationResult,
    VerificationTask,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATHS = {
    "researcher": Path("prompts/researcher_v1.txt"),
    "analyzer": Path("prompts/analyzer_v1.txt"),
    "verifier": Path("prompts/verifier_v1.txt"),
}

FALLBACK_PROMPTS = {
    "researcher": """You are the ARGUS Researcher.

Your job is to gather factual background, constraints, references, and unknowns.
Do not provide final conclusions or implementation advice.
Return only valid JSON with keys:
- facts
- constraints
- references
- unknowns
- confidence
""",
    "analyzer": """You are the ARGUS Analyzer.

Your job is to produce solution logic, assumptions, tradeoffs, risks, and validation checks.
Do not provide broad background research or the final user-facing conclusion.
Return only valid JSON with keys:
- proposed_solution
- assumptions
- tradeoffs
- risks
- validation_checks
""",
    "verifier": """You are the ARGUS Verifier.

Your job is to independently pressure-test the task from the same shared task snapshot.
Focus on risks, hidden assumptions, edge cases, and required validation.
Do not produce the final answer or the primary implementation plan.
Return only valid JSON with keys:
- critical_risks
- hidden_assumptions
- edge_cases
- validation_requirements
- confidence
""",
}

DEFAULT_CONFIG: dict[str, ConnectorConfig] = {
    "researcher": ConnectorConfig(max_tokens=1200, temperature=0.2),
    "analyzer": ConnectorConfig(max_tokens=1600, temperature=0.2),
    "verifier": ConnectorConfig(max_tokens=1200, temperature=0.2),
}


def _load_role_prompt(role: str) -> str:
    prompt_path = PROMPT_PATHS[role]
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": f"{prompt_path.name} not found, using inline fallback"})
        return FALLBACK_PROMPTS[role]


def _build_role_sub_query(shared_state: SharedTaskState, task: BaseModel, role: str) -> str:
    role_title = role.capitalize()
    shared_payload = shared_state.model_dump_json(indent=2, exclude_none=True)
    task_payload = task.model_dump_json(indent=2, exclude_none=True)
    return (
        f"{_load_role_prompt(role)}\n\n"
        f"Shared task state:\n{shared_payload}\n\n"
        f"{role_title} task:\n{task_payload}\n\n"
        "Return only valid JSON."
    )


ResultT = TypeVar("ResultT", bound=BaseModel)


@dataclass
class WorkerOutcome(Generic[ResultT]):
    """Parsed role result plus the underlying connector response.

    Exposes the connector's real token usage and per-call latency so the
    API layer can report them instead of stub values.
    """

    result: ResultT
    response: ConnectorResponse


class RoleTaskError(Exception):
    """Worker failure that preserves the underlying ConnectorResponse.

    Raised on non-success connector replies so the parallel path can
    surface the TRUE provider status (rate_limited, timeout, ...) and
    retry hints instead of collapsing everything to a generic error.
    """

    def __init__(self, role: str, response: ConnectorResponse):
        self.role = role
        self.response = response
        detail = response.error or response.status.value
        if response.status == ConnectorStatus.RATE_LIMITED:
            wait = f"retry_after_s={response.retry_after_s}" if response.retry_after_s else "no retry hint"
            detail = f"provider rate limited ({wait}): {detail}"
        super().__init__(f"{role.capitalize()} task failed: {detail}")


async def _query_with_retry(
    connector: BaseConnector,
    prompt: str,
    sub_query: str,
    config: ConnectorConfig,
) -> ConnectorResponse:
    """Run a connector query with one automatic retry on timeout."""
    response = await connector.query(prompt, sub_query, config)

    if response.status == ConnectorStatus.TIMEOUT and config.max_retries > 0:
        logger.info({
            "message": "Retrying connector after timeout",
            "connector_id": connector.connector_id,
        })
        retry_config = ConnectorConfig(
            timeout_s=config.timeout_s,
            max_retries=0,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        response = await connector.query(prompt, sub_query, retry_config)

    return response


def _parse_role_result(raw: str, result_model: type[ResultT]) -> ResultT:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return result_model.model_validate(json.loads(raw.strip()))


async def _run_role_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: BaseModel,
    role: str,
    result_model: type[ResultT],
    config: ConnectorConfig | None = None,
) -> WorkerOutcome[ResultT]:
    if config is None:
        config = replace(DEFAULT_CONFIG[role])

    response = await _query_with_retry(
        connector=connector,
        prompt=shared_state.original_query,
        sub_query=_build_role_sub_query(shared_state, task, role),
        config=config,
    )

    if response.status != ConnectorStatus.SUCCESS or not response.content:
        raise RoleTaskError(role=role, response=response)

    result = _parse_role_result(response.content, result_model)
    return WorkerOutcome(result=result, response=response)


async def run_research_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: ResearchTask,
    config: ConnectorConfig | None = None,
) -> WorkerOutcome[ResearchResult]:
    return await _run_role_task(
        connector=connector,
        shared_state=shared_state,
        task=task,
        role="researcher",
        result_model=ResearchResult,
        config=config,
    )


async def run_analysis_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: AnalysisTask,
    config: ConnectorConfig | None = None,
) -> WorkerOutcome[AnalysisResult]:
    return await _run_role_task(
        connector=connector,
        shared_state=shared_state,
        task=task,
        role="analyzer",
        result_model=AnalysisResult,
        config=config,
    )


async def run_verification_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: VerificationTask,
    config: ConnectorConfig | None = None,
) -> WorkerOutcome[VerificationResult]:
    return await _run_role_task(
        connector=connector,
        shared_state=shared_state,
        task=task,
        role="verifier",
        result_model=VerificationResult,
        config=config,
    )
