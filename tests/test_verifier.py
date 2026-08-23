import pytest

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage
from app.orchestration.contracts import SharedTaskState, VerificationTask
from app.orchestration.workers import run_verification_task


class VerificationConnector(BaseConnector):
    connector_id = "verification-mock"
    display_name = "Verification Mock"
    capabilities = ["text", "verification"]
    is_available = True

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id="verification-mock-model",
            content=(
                '{"critical_risks":["r1"],"hidden_assumptions":["a1"],'
                '"edge_cases":["e1"],"validation_requirements":["v1"],"confidence":"high"}'
            ),
            latency_ms=10,
            token_usage=TokenUsage(11, 7, 18),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_run_verification_task_parses_json():
    outcome = await run_verification_task(
        connector=VerificationConnector(),
        shared_state=SharedTaskState(
            request_id="req-1",
            original_query="Design a secure workflow",
            main_objective="Produce a robust solution",
            expected_final_output="json",
        ),
        task=VerificationTask(
            objective="Pressure-test the task and solution space",
            scope=["risks", "edge cases"],
            do_not_cover=["implementation", "final answer"],
            required_output_fields=[
                "critical_risks",
                "hidden_assumptions",
                "edge_cases",
                "validation_requirements",
                "confidence",
            ],
        ),
        config=ConnectorConfig(),
    )
    assert outcome.result.critical_risks == ["r1"]
    assert outcome.result.confidence == "high"
    assert outcome.response.token_usage.total_tokens == 18
