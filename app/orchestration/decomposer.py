from app.orchestration.contracts import (
    AnalysisTask,
    OrchestrationPlan,
    ResearchTask,
    SharedTaskState,
    VerificationTask,
)

SHORT_CIRCUIT_WORD_THRESHOLD = 50
MAX_OBJECTIVE_WORDS = 24


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


def _clean_query_text(query: str) -> str:
    return " ".join(query.split()).strip()


def _derive_main_objective(query: str) -> str:
    words = _clean_query_text(query).split()
    if len(words) <= MAX_OBJECTIVE_WORDS:
        return " ".join(words)
    return " ".join(words[:MAX_OBJECTIVE_WORDS]).strip() + "..."


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
        objective=(
            "Pressure-test the task and likely solution space for risks, "
            "hidden assumptions, and edge cases."
        ),
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
