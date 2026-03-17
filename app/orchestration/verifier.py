import json
from pathlib import Path

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.orchestration.contracts import SharedTaskState, VerificationResult, VerificationTask
from app.utils.logger import get_logger

logger = get_logger(__name__)

VERIFIER_PROMPT_PATH = Path("prompts/verifier_v1.txt")

VERIFIER_FALLBACK_PROMPT = """You are the ARGUS Verifier.

Your job is to independently pressure-test the task from the same shared task snapshot.
Focus on risks, hidden assumptions, edge cases, and required validation.
Do not produce the final answer or the primary implementation plan.
Return only valid JSON with keys:
- critical_risks
- hidden_assumptions
- edge_cases
- validation_requirements
- confidence
"""


def _load_verifier_prompt() -> str:
    try:
        return VERIFIER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": "verifier_v1.txt not found, using inline fallback"})
        return VERIFIER_FALLBACK_PROMPT


def _build_verifier_sub_query(shared_state: SharedTaskState, task: VerificationTask) -> str:
    return (
        f"{_load_verifier_prompt()}\n\n"
        f"Shared task state:\n{shared_state.model_dump_json(indent=2)}\n\n"
        f"Verification task:\n{task.model_dump_json(indent=2)}\n\n"
        "Return only valid JSON."
    )


def _parse_verification_result(raw: str) -> VerificationResult:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return VerificationResult.model_validate(json.loads(raw.strip()))


async def run_verification_task(
    connector: BaseConnector,
    shared_state: SharedTaskState,
    task: VerificationTask,
    config: ConnectorConfig | None = None,
) -> VerificationResult:
    if config is None:
        config = ConnectorConfig(max_tokens=1200, temperature=0.2)

    response = await connector.query(
        prompt=shared_state.original_query,
        sub_query=_build_verifier_sub_query(shared_state, task),
        config=config,
    )

    if response.status != ConnectorStatus.SUCCESS or not response.content:
        raise ValueError(f"Verification task failed: {response.error or response.status.value}")

    return _parse_verification_result(response.content)
