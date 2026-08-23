import pytest

from app.connectors.base import ConnectorResponse, ConnectorStatus, TokenUsage
from app.orchestration.report_planner import _fallback_plan, build_report_plan


class ScriptedPlanner:
    connector_id = "scripted-planner"

    def __init__(self, content: str | None = None, status: ConnectorStatus = ConnectorStatus.SUCCESS):
        self._content = content
        self._status = status

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id="scripted",
            content=self._content or "",
            latency_ms=1,
            token_usage=TokenUsage(),
            status=self._status,
            sub_query=sub_query,
        )


@pytest.mark.asyncio
async def test_build_report_plan_parses_subtasks():
    content = (
        '{"report_title": "RAG vs Long Context", "subtasks": ['
        '{"subtask_id": "s1", "title": "Retrieval", "objective": "Assess RAG", "focus": ["latency"]},'
        '{"subtask_id": "s2", "title": "Context", "objective": "Assess long context"}'
        "]}"
    )
    plan, planner_id = await build_report_plan("Compare RAG and long context", "req-1", ScriptedPlanner(content))
    assert planner_id == "scripted-planner"
    assert plan.report_title == "RAG vs Long Context"
    assert [s.subtask_id for s in plan.subtasks] == ["s1", "s2"]
    assert plan.shared_state.original_query == "Compare RAG and long context"


@pytest.mark.asyncio
async def test_build_report_plan_clamps_to_max_subtasks():
    raw = ",".join(
        f'{{"subtask_id": "s{i}", "title": "t{i}", "objective": "o{i}"}}' for i in range(1, 8)
    )
    content = f'{{"report_title": "Big", "subtasks": [{raw}]}}'
    plan, _ = await build_report_plan("big query", "req-2", ScriptedPlanner(content))
    assert len(plan.subtasks) == 5


@pytest.mark.asyncio
async def test_build_report_plan_falls_back_on_bad_json():
    plan, _ = await build_report_plan("some query", "req-3", ScriptedPlanner("not json"))
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].subtask_id == "s1"
    assert plan.subtasks[0].objective == "some query"


@pytest.mark.asyncio
async def test_build_report_plan_falls_back_on_error_status():
    plan, _ = await build_report_plan(
        "some query", "req-4", ScriptedPlanner(status=ConnectorStatus.ERROR)
    )
    fallback = _fallback_plan("some query", "req-4")
    assert plan == fallback
