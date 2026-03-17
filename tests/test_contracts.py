from app.orchestration.contracts import (
    AggregationInput,
    AnalysisResult,
    AnalysisTask,
    ResearchResult,
    ResearchTask,
    SharedTaskState,
    VerificationResult,
    VerificationTask,
)


def test_shared_task_state_shape():
    state = SharedTaskState(
        request_id="req-1",
        original_query="Build a secure scraping pipeline",
        main_objective="Produce a safe and practical design",
        task_context=["backend service"],
        constraints=["respect robots.txt"],
        global_rules=["no final answer"],
        expected_final_output="technical recommendation",
    )
    assert state.request_id == "req-1"
    assert state.constraints == ["respect robots.txt"]


def test_role_tasks_default_roles():
    research_task = ResearchTask(
        objective="Collect facts",
        scope=["laws", "rate limits"],
        do_not_cover=["implementation"],
        required_output_fields=["facts", "references"],
    )
    analysis_task = AnalysisTask(
        objective="Design the implementation",
        scope=["pipeline", "error handling"],
        do_not_cover=["broad background"],
        required_output_fields=["proposed_solution", "risks"],
    )
    verification_task = VerificationTask(
        objective="Pressure-test the solution space",
        scope=["risks", "edge cases"],
        do_not_cover=["implementation", "final answer"],
        required_output_fields=["critical_risks", "edge_cases"],
    )
    assert research_task.role == "researcher"
    assert analysis_task.role == "analyzer"
    assert verification_task.role == "verifier"


def test_aggregation_input_wraps_results():
    aggregation_input = AggregationInput(
        shared_state=SharedTaskState(
            request_id="req-2",
            original_query="test",
            main_objective="test objective",
            expected_final_output="markdown",
        ),
        research_result=ResearchResult(
            facts=["f1"],
            constraints=["c1"],
            references=["r1"],
            unknowns=["u1"],
            confidence="high",
        ),
        analysis_result=AnalysisResult(
            proposed_solution="Do X",
            assumptions=["a1"],
            tradeoffs=["t1"],
            risks=["r1"],
            validation_checks=["v1"],
        ),
        verification_result=VerificationResult(
            critical_risks=["r1"],
            hidden_assumptions=["a1"],
            edge_cases=["e1"],
            validation_requirements=["v1"],
            confidence="medium",
        ),
    )
    assert aggregation_input.research_result.confidence == "high"
    assert aggregation_input.analysis_result.proposed_solution == "Do X"
    assert aggregation_input.verification_result.edge_cases == ["e1"]
