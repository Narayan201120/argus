import time
from pathlib import Path

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYNTHESIS_PROMPT_PATH = Path("prompts/synthesis_v1.txt")

FALLBACK_PROMPT = (
    "You are a synthesis AI. You received the original user query and labeled responses "
    "from multiple AI models, each addressing a different aspect of the query. "
    "Synthesize these into one coherent, non-redundant, authoritative answer."
)


def _load_synthesis_prompt() -> str:
    try:
        return SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": "synthesis_v1.txt not found, using inline fallback"})
        return FALLBACK_PROMPT


def _build_synthesis_prompt(
    original_query: str,
    response_bundle: dict[str, ConnectorResponse],
    system_prompt: str,
) -> str:
    sections = []
    for connector_id, response in response_bundle.items():
        if response.status == ConnectorStatus.SUCCESS and response.content:
            task_label = response.sub_query or "general"
            sections.append(
                f"--- [{connector_id.upper()}] (Task: {task_label}) ---\n{response.content}"
            )

    assembled = "\n\n".join(sections)
    return (
        f"{system_prompt}\n\n"
        f"Original user query: {original_query}\n\n"
        f"Model responses:\n{assembled}\n\n"
        f"Now provide the single unified answer:"
    )


def _labeled_concat_fallback(
    response_bundle: dict[str, ConnectorResponse],
) -> str:
    """Last resort: concatenate successful responses with labels."""
    lines = [
        "The following responses were collected from multiple AI models. "
        "Each model addressed a specific aspect of your query:\n"
    ]
    for connector_id, response in response_bundle.items():
        if response.status == ConnectorStatus.SUCCESS and response.content:
            lines.append(f"**{connector_id.upper()}:**\n{response.content}\n")

    if len(lines) == 1:
        return "No successful responses were collected from any connector."
    return "\n".join(lines)


async def synthesize(
    original_query: str,
    response_bundle: dict[str, ConnectorResponse],
    synthesizer_chain: list[BaseConnector],
    config: ConnectorConfig | None = None,
) -> tuple[str, str, ConnectorResponse | None]:
    """
    Synthesize connector responses into a single unified answer.

    Tries each synthesizer in the fallback chain until one succeeds.
    Falls back to labeled concatenation if all synthesizers fail.

    Args:
        original_query: The original user query
        response_bundle: Dict of connector_id -> ConnectorResponse
        synthesizer_chain: Ordered list of synthesizer connectors to try
        config: Config for the synthesis LLM call

    Returns:
        Tuple of (synthesized_content, synthesizer_id_used, raw_response_or_None)
    """
    if config is None:
        config = ConnectorConfig(max_tokens=4096, temperature=0.3)

    successful = {
        cid: r
        for cid, r in response_bundle.items()
        if r.status == ConnectorStatus.SUCCESS and r.content
    }

    # No connector responses at all
    if not successful:
        logger.warning({"message": "All connectors failed — returning diagnostic response"})
        return (
            "All configured connectors failed to respond. "
            "Please check your API keys and connector availability.",
            "system",
            None,
        )

    # Only one response — no synthesis needed, return directly
    if len(successful) == 1:
        cid, r = next(iter(successful.items()))
        return r.content, cid, r

    system_prompt = _load_synthesis_prompt()
    synthesis_input = _build_synthesis_prompt(original_query, successful, system_prompt)

    for synthesizer in synthesizer_chain:
        if not synthesizer.is_available:
            logger.info({
                "message": "Synthesizer unavailable, trying next",
                "synthesizer": synthesizer.connector_id,
            })
            continue

        try:
            logger.info({"message": "Attempting synthesis", "synthesizer": synthesizer.connector_id})
            response = await synthesizer.query(
                prompt="You are the synthesis layer of an AI orchestration system.",
                sub_query=synthesis_input,
                config=config,
            )

            if response.status == ConnectorStatus.SUCCESS and response.content:
                logger.info({
                    "message": "Synthesis succeeded",
                    "synthesizer": synthesizer.connector_id,
                    "latency_ms": response.latency_ms,
                })
                return response.content, synthesizer.connector_id, response

            logger.warning({
                "message": "Synthesizer responded with non-success",
                "synthesizer": synthesizer.connector_id,
                "status": response.status,
            })

        except Exception as e:
            logger.error({
                "message": "Synthesizer exception",
                "synthesizer": synthesizer.connector_id,
                "error": str(e),
            })
            continue

    # All synthesizers failed — use labeled concatenation as last resort
    logger.warning({"message": "All synthesizers exhausted — using labeled concatenation fallback"})
    content = _labeled_concat_fallback(successful)
    return content, "fallback_concat", None
