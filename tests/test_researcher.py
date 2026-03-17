import pytest

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.orchestration.contracts import ResearchTask, SharedTaskState
from app.orchestration.researcher import run_research_task


class ResearchConnector(BaseConnector):
    connector_id = "research-mock"
    display_name = "Research Mock"
    capabilities = ["text", "research"]
    is_available = True

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id="research-mock-model",
            content=(
                '{"facts":["f1"],"constraints":["c1"],"references":["r1"],'
                '"unknowns":["u1"],"confidence":"high"}'
            ),
            latency_ms=10,
            token_usage=TokenUsage(),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_run_research_task_parses_json():
    result = await run_research_task(
        connector=ResearchConnector(),
        shared_state=SharedTaskState(
            request_id="req-1",
            original_query="Find constraints",
            main_objective="Collect research",
            expected_final_output="json",
        ),
        task=ResearchTask(
            objective="Gather facts and constraints",
            scope=["facts", "constraints"],
            do_not_cover=["implementation"],
            required_output_fields=["facts", "constraints", "references", "unknowns", "confidence"],
        ),
        config=ConnectorConfig(),
    )
    assert result.facts == ["f1"]
    assert result.confidence == "high"
