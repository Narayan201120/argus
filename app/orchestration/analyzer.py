import json
from pathlib import Path

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.orchestration.contracts import AnalysisResult, AnalysisTask, SharedTaskState
from app.utils.logger import get_logger

logger = get_logger(__name__)

ANALYZER_PROMPT_PATH = Path("prompts/analyzer_v1.txt")

ANALYZER_FALLBACK_PROMPT = """You are the ARGUS Analyzer.

Your job is to produce solution logic, assumptions, tradeoffs, risks, and validation checks.
Do not provide broad background research or the final user-facing conclusion.
Return only valid JSON with keys:
- proposed_solution
- assumptions
- tradeoffs
- risks
- validation_checks
"""


def _load_analyzer_prompt() -> str:
    try:
        return ANALYZER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": "analyzer_v1.txt not found, using inline fallback"})
        return ANALYZER_FALLBACK_PROMPT


def _build_analyzer_sub_query(shared_state: SharedTaskState, task: AnalysisTask) -> str:
    return (
        f"{_load_analyzer_prompt()}\n\n"
        f"Shared task state:\n{shared_state.model_dump_json(indent=2)}\n\n"
        f"Analysis task:\n{task.model_dump_json(indent=2)}\n\n"
        "Return only valid JSON."
    )


def _parse_analysis_result(raw: str) -> AnalysisResult:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return AnalysisResult.model_validate(json.loads(raw.strip()))


async def run_analysis_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: AnalysisTask,
    config: ConnectorConfig | None = None,
) -> AnalysisResult:
    if config is None:
        config = ConnectorConfig(max_tokens=1600, temperature=0.2)

    response = await connector.query(
        prompt=shared_state.original_query,
        sub_query=_build_analyzer_sub_query(shared_state, task),
        config=config,
    )

    if response.status != ConnectorStatus.SUCCESS or not response.content:
        raise ValueError(f"Analysis task failed: {response.error or response.status.value}")

    return _parse_analysis_result(response.content)
