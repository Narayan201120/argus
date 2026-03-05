import json
import asyncio
from pathlib import Path

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.utils.logger import get_logger

logger = get_logger(__name__)

SHORT_CIRCUIT_WORD_THRESHOLD = 50

DECOMPOSER_SYSTEM_PROMPT = """You are a query planner for ARGUS, an AI orchestration system.

You will receive a user query and a list of AI connectors with their capabilities.

Your job: decompose the query into specific, complementary sub-queries — one per connector.

Rules:
- Each sub-query must cover a DIFFERENT aspect of the overall question
- Assign sub-queries based on each connector's capabilities (e.g. code tasks to code-capable connectors)
- Sub-queries must be self-contained and answerable independently
- Return ONLY valid JSON in exactly this format, no extra text:
{
  "connector_id_1": "sub-query for connector 1",
  "connector_id_2": "sub-query for connector 2"
}"""


def _is_simple_query(query: str) -> bool:
    """Heuristic: short, single-intent queries bypass decomposition."""
    word_count = len(query.split())
    has_multiple_questions = query.count("?") > 1
    has_multiple_lines = query.count("\n") > 2
    return (
        word_count < SHORT_CIRCUIT_WORD_THRESHOLD
        and not has_multiple_questions
        and not has_multiple_lines
    )


def _parse_json_response(raw: str) -> dict[str, str]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


async def decompose_query(
    query: str,
    connectors: list[BaseConnector],
    decomposer_connector: BaseConnector | None = None,
) -> dict[str, str] | None:
    """
    Decompose a user query into per-connector sub-queries.

    Returns:
        dict[connector_id -> sub_query] if decomposition succeeded,
        or None to signal short-circuit (aggregator answers directly).
    """
    if not connectors:
        return None

    if len(connectors) == 1:
        # Only one connector — give it the whole query
        return {connectors[0].connector_id: query}

    if _is_simple_query(query):
        logger.info({"message": "Short-circuit: simple query, aggregator responds directly"})
        return None

    # Build capability context for the decomposer
    connector_profiles = "\n".join(
        f"- {c.connector_id}: capabilities = [{', '.join(c.capabilities)}]"
        for c in connectors
    )

    decomposer_prompt = (
        f"User query: {query}\n\n"
        f"Available connectors:\n{connector_profiles}\n\n"
        f"Return the JSON sub-query assignments now."
    )

    dc = decomposer_connector or connectors[0]

    try:
        config = ConnectorConfig(timeout_s=20, max_tokens=512, temperature=0.1)
        response = await dc.query(
            prompt=DECOMPOSER_SYSTEM_PROMPT,
            sub_query=decomposer_prompt,
            config=config,
        )

        if response.status != ConnectorStatus.SUCCESS or not response.content:
            logger.warning({
                "message": "Decomposer failed, falling back to full query per connector",
                "status": response.status,
            })
            return {c.connector_id: query for c in connectors}

        sub_queries = _parse_json_response(response.content)

        # Ensure every active connector has a sub-query (fill missing with full query)
        connector_ids = {c.connector_id for c in connectors}
        for cid in connector_ids:
            if cid not in sub_queries:
                sub_queries[cid] = query

        logger.info({
            "message": "Query decomposed",
            "num_sub_queries": len(sub_queries),
            "connectors": list(sub_queries.keys()),
        })
        return sub_queries

    except (json.JSONDecodeError, Exception) as e:
        logger.error({"message": "Decomposer error, falling back", "error": str(e)})
        return {c.connector_id: query for c in connectors}
