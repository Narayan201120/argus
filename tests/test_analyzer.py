import pytest

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.orchestration.contracts import AnalysisTask, SharedTaskState
from app.orchestration.workers import run_analysis_task


class AnalysisConnector(BaseConnector):
    connector_id = "analysis-mock"
    display_name = "Analysis Mock"
    capabilities = ["text", "analysis"]
    is_available = True

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id="analysis-mock-model",
            content=(
                '{"proposed_solution":"Do X","assumptions":["a1"],'
                '"tradeoffs":["t1"],"risks":["r1"],"validation_checks":["v1"]}'
            ),
            latency_ms=10,
            token_usage=TokenUsage(),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_run_analysis_task_parses_json():
    result = await run_analysis_task(
        connector=AnalysisConnector(),
        shared_state=SharedTaskState(
            request_id="req-1",
            original_query="Design the system",
            main_objective="Produce a technical solution",
            expected_final_output="json",
        ),
        task=AnalysisTask(
            objective="Produce solution logic",
            scope=["architecture", "tradeoffs"],
            do_not_cover=["broad research"],
            required_output_fields=["proposed_solution", "assumptions", "tradeoffs", "risks", "validation_checks"],
        ),
        config=ConnectorConfig(),
    )
    assert result.proposed_solution == "Do X"
    assert result.validation_checks == ["v1"]
