"""Planner for the deep-report pipeline.

Splits a query into a small set of complementary subtasks. Falls back to a
single whole-query subtask when the planner call or its JSON is unusable,
so a report never hard-fails at planning time.
"""

import json

from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.orchestration.contracts import SharedTaskState
from app.orchestration.report_contracts import MAX_REPORT_SUBTASKS, ReportPlan, ReportSubtask
from app.utils.logger import get_logger

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the ARGUS report planner.

Split the user's research request into 2 to 5 complementary subtasks that
together cover the request. Each subtask must be independently researchable.

Return ONLY valid JSON in exactly this shape:
{
  "report_title": "short report title",
  "subtasks": [
    {"subtask_id": "s1", "title": "short title",
     "objective": "what to investigate", "focus": ["aspect one"]}
  ]
}"""


def _parse_json_payload(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _fallback_plan(query: str, request_id: str) -> ReportPlan:
    return ReportPlan(
        report_title=query[:80],
        shared_state=SharedTaskState(
            request_id=request_id,
            original_query=query,
            main_objective=query,
            expected_final_output="A structured Markdown research report.",
        ),
        subtasks=[
            ReportSubtask(subtask_id="s1", title="Full investigation", objective=query)
        ],
    )


async def build_report_plan(
    query: str,
    request_id: str,
    planner: BaseConnector,
    config: ConnectorConfig | None = None,
) -> tuple[ReportPlan, str]:
    """Returns (plan, planner_connector_id). Never raises for planner errors."""
    fallback = _fallback_plan(query, request_id)

    try:
        response = await planner.query(
            prompt=PLANNER_SYSTEM_PROMPT,
            sub_query=(
                f"User request: {query}\n\n"
                f"Produce between 2 and {MAX_REPORT_SUBTASKS} subtasks."
            ),
            config=config or ConnectorConfig(max_tokens=1024, temperature=0.2),
        )
        if response.status != ConnectorStatus.SUCCESS or not response.content:
            logger.warning({"message": "Report planner non-success, using fallback"})
            return fallback, planner.connector_id
        payload = _parse_json_payload(response.content)
        raw_subtasks = payload.get("subtasks") or []
        subtasks = [
            ReportSubtask.model_validate(item)
            for item in raw_subtasks[:MAX_REPORT_SUBTASKS]
        ]
        if len(subtasks) == 0:
            return fallback, planner.connector_id

        plan = ReportPlan(
            report_title=str(payload.get("report_title") or query[:80]),
            shared_state=SharedTaskState(
                request_id=request_id,
                original_query=query,
                main_objective=str(payload.get("report_title") or query[:80]),
                expected_final_output="A structured Markdown research report.",
            ),
            subtasks=subtasks,
        )
        return plan, planner.connector_id
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning({
            "message": "Report planner parse failure, using fallback",
            "error": str(exc),
        })
        return fallback, planner.connector_id
