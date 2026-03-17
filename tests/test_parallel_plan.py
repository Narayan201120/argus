from app.orchestration.decomposer import build_parallel_plan


def test_build_parallel_plan_returns_role_scoped_tasks():
    plan = build_parallel_plan(
        query="Design a safe scraping pipeline",
        request_id="req-123",
    )

    assert plan.shared_state.request_id == "req-123"
    assert plan.research_task.role == "researcher"
    assert plan.analysis_task.role == "analyzer"
    assert plan.verification_task.role == "verifier"
    assert "Do not produce the final answer." in plan.analysis_task.do_not_cover
