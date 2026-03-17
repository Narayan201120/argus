import json

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.orchestration.contracts import (
    AnalysisTask,
    OrchestrationPlan,
    ResearchTask,
    SharedTaskState,
    VerificationTask,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

SHORT_CIRCUIT_WORD_THRESHOLD = 50
MAX_OBJECTIVE_WORDS = 24

DECOMPOSER_SYSTEM_PROMPT = """You are a query planner for ARGUS, an AI orchestration system.

You will receive a user query and a list of AI connectors with their capabilities.

Your job: decompose the query into specific, complementary sub-queries - one per connector.

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


def _clean_query_text(query: str) -> str:
    return " ".join(query.split()).strip()


def _derive_main_objective(query: str) -> str:
    words = _clean_query_text(query).split()
    if len(words) <= MAX_OBJECTIVE_WORDS:
        return " ".join(words)
    return " ".join(words[:MAX_OBJECTIVE_WORDS]).strip() + "..."


async def decompose_query(
    query: str,
    connectors: list[BaseConnector],
    decomposer_connector: BaseConnector | None = None,
) -> dict[str, str] | None:
    """
    Legacy decomposition path retained for compatibility with existing tests and call sites.
    """
    if not connectors:
        return None

    if len(connectors) == 1:
        return {connectors[0].connector_id: query}

    if _is_simple_query(query):
        logger.info({"message": "Short-circuit: simple query, aggregator responds directly"})
        return None

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


def build_parallel_plan(
    query: str,
    request_id: str,
) -> OrchestrationPlan:
    normalized_query = _clean_query_text(query)

    shared_state = SharedTaskState(
        request_id=request_id,
        original_query=normalized_query,
        main_objective=_derive_main_objective(normalized_query),
        task_context=[
            "Parallel orchestration: researcher and analyzer run from the same frozen task snapshot.",
            "Aggregator reconciles role-scoped outputs into the final response.",
        ],
        constraints=[
            "Researcher, analyzer, and verifier must not depend on each other's live outputs.",
            "Workers must stay within assigned scope and avoid writing the final answer.",
        ],
        global_rules=[
            "Return only role-scoped output.",
            "State uncertainty explicitly instead of inventing facts.",
            "Do not silently cross role boundaries.",
        ],
        expected_final_output="A reconciled response grounded in research and analysis outputs.",
    )

    research_task = ResearchTask(
        objective="Collect factual background, constraints, references, and unknowns relevant to the query.",
        scope=[
            "Relevant facts and background",
            "Operational or domain constraints",
            "References or source leads",
            "Unknowns that could affect confidence",
        ],
        do_not_cover=[
            "Do not propose the final implementation or recommendation.",
            "Do not produce the final answer.",
        ],
        required_output_fields=["facts", "constraints", "references", "unknowns", "confidence"],
    )

    analysis_task = AnalysisTask(
        objective="Develop the technical or logical solution path from the same shared task snapshot.",
        scope=[
            "Solution logic or implementation path",
            "Assumptions required to proceed",
            "Tradeoffs and risks",
            "Validation checks for the proposed approach",
        ],
        do_not_cover=[
            "Do not produce broad background research.",
            "Do not claim unsupported facts as certain.",
            "Do not produce the final answer.",
        ],
        required_output_fields=[
            "proposed_solution",
            "assumptions",
            "tradeoffs",
            "risks",
            "validation_checks",
        ],
    )

    verification_task = VerificationTask(
        objective="Pressure-test the task and likely solution space for risks, hidden assumptions, and edge cases.",
        scope=[
            "Critical risks that could degrade the final answer",
            "Hidden assumptions that need to be surfaced",
            "Edge cases and failure scenarios",
            "Validation requirements before strong confidence is justified",
        ],
        do_not_cover=[
            "Do not produce the primary implementation plan.",
            "Do not produce broad background research.",
            "Do not produce the final answer.",
        ],
        required_output_fields=[
            "critical_risks",
            "hidden_assumptions",
            "edge_cases",
            "validation_requirements",
            "confidence",
        ],
    )

    return OrchestrationPlan(
        shared_state=shared_state,
        research_task=research_task,
        analysis_task=analysis_task,
        verification_task=verification_task,
    )
