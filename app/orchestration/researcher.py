import json
from pathlib import Path

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.orchestration.contracts import ResearchResult, ResearchTask, SharedTaskState
from app.utils.logger import get_logger

logger = get_logger(__name__)

RESEARCHER_PROMPT_PATH = Path("prompts/researcher_v1.txt")

RESEARCHER_FALLBACK_PROMPT = """You are the ARGUS Researcher.

Your job is to gather factual background, constraints, references, and unknowns.
Do not provide final conclusions or implementation advice.
Return only valid JSON with keys:
- facts
- constraints
- references
- unknowns
- confidence
"""


def _load_researcher_prompt() -> str:
    try:
        return RESEARCHER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": "researcher_v1.txt not found, using inline fallback"})
        return RESEARCHER_FALLBACK_PROMPT


def _build_researcher_sub_query(shared_state: SharedTaskState, task: ResearchTask) -> str:
    return (
        f"{_load_researcher_prompt()}\n\n"
        f"Shared task state:\n{shared_state.model_dump_json(indent=2)}\n\n"
        f"Research task:\n{task.model_dump_json(indent=2)}\n\n"
        "Return only valid JSON."
    )


def _parse_research_result(raw: str) -> ResearchResult:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return ResearchResult.model_validate(json.loads(raw.strip()))


async def run_research_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: ResearchTask,
    config: ConnectorConfig | None = None,
) -> ResearchResult:
    if config is None:
        config = ConnectorConfig(max_tokens=1200, temperature=0.2)

    response = await connector.query(
        prompt=shared_state.original_query,
        sub_query=_build_researcher_sub_query(shared_state, task),
        config=config,
    )

    if response.status != ConnectorStatus.SUCCESS or not response.content:
        raise ValueError(f"Research task failed: {response.error or response.status.value}")

    return _parse_research_result(response.content)
