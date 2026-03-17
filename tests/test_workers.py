from app.orchestration.contracts import ResearchTask, SharedTaskState
from app.orchestration.workers import _build_role_sub_query


def test_worker_prompt_uses_clean_shared_state_and_task():
    prompt = _build_role_sub_query(
        SharedTaskState(
            request_id=" req-1 ",
            original_query="  Explain   the   system ",
            main_objective=" Explain the system ",
            task_context=["  backend service  ", "backend service"],
            constraints=["  avoid noise  "],
            global_rules=["  no final answer  "],
            expected_final_output=" markdown ",
        ),
        ResearchTask(
            objective="  Gather   facts  ",
            scope=["  facts  ", "facts"],
            do_not_cover=["  implementation  "],
            required_output_fields=[" facts ", "facts"],
        ),
        "researcher",
    )
    assert "Explain the system" in prompt
    assert "backend service" in prompt
    assert prompt.count("backend service") == 1
    assert prompt.count('"implementation"') == 1
