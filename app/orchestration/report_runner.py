"""Deep-report pipeline executor.

Runs: plan subtasks -> bounded parallel research/analysis tracks ->
global verification -> writer -> bounded reviewer repair loop. Updates the
job store at every phase transition; a report completes even when some
tracks or the reviewer fail, and only a total writer failure marks the job
failed (with a deterministic labeled fallback as last resort).
"""

import asyncio
import json
from pathlib import Path

from app.config import settings
from app.connectors.base import BaseConnector, ConnectorConfig, ConnectorStatus
from app.orchestration.binding import binding_service
from app.orchestration.contracts import (
    AnalysisTask,
    ResearchTask,
    SharedTaskState,
    VerificationTask,
)
from app.orchestration.report_contracts import (
    ReviewVerdict,
    TrackResult,
    VerificationSummary,
)
from app.orchestration.report_jobs import report_job_store
from app.orchestration.report_planner import build_report_plan
from app.orchestration.workers import (
    run_analysis_task,
    run_research_task,
    run_verification_task,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

TRACK_CONCURRENCY = 3

WRITER_PROMPT_PATH = Path("prompts/writer_v1.txt")
REVIEWER_PROMPT_PATH = Path("prompts/reviewer_v1.txt")

FALLBACK_WRITER_PROMPT = (
    "You are the ARGUS report writer. Write a complete Markdown research "
    "report grounded strictly in the provided track outputs and verification "
    "summary. Include an executive summary, one section per subtask, and a "
    "closing risks/open-questions section."
)

FALLBACK_REVIEWER_PROMPT = (
    "You are the ARGUS report reviewer. Judge whether the draft report is "
    "publishable given its track outputs. Return only valid JSON: "
    '{"approved": true, "issues": []}'
)


def _load_prompt(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": f"{path.name} not found, using inline fallback"})
        return fallback


def _parse_json_payload(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


async def execute_report(
    job_id: str,
    query: str,
    config: ConnectorConfig,
    overrides: dict[str, list[str]] | None = None,
    active_connectors: list[BaseConnector] | None = None,
) -> None:
    """Entry point scheduled by the API route. Never raises."""
    request_id = job_id
    try:
        await _execute(job_id, query, request_id, config, overrides, active_connectors or [])
    except Exception as exc:
        logger.error({
            "message": "Report execution failed",
            "job_id": job_id,
            "error": str(exc),
        })
        await report_job_store.update(job_id, status="failed", error=str(exc))


async def _execute(
    job_id: str,
    query: str,
    request_id: str,
    config: ConnectorConfig,
    overrides: dict[str, list[str]] | None,
    active_connectors: list[BaseConnector],
) -> None:
    if not active_connectors:
        raise ValueError("No active connectors supplied to report runner")

    await report_job_store.update(job_id, status="running", progress="Planning subtasks")
    planner = binding_service.select_connector(active_connectors, "planner", overrides=overrides)
    plan, planner_id = await build_report_plan(query, request_id, planner, config)
    assignments: dict[str, str] = {"planner": planner_id}
    logger.info({
        "message": "Report planned",
        "job_id": job_id,
        "subtasks": [s.subtask_id for s in plan.subtasks],
    })

    await report_job_store.update(
        job_id,
        progress=f"Running {len(plan.subtasks)} research tracks",
        role_assignments=assignments,
    )
    semaphore = asyncio.Semaphore(TRACK_CONCURRENCY)
    tracks = list(await asyncio.gather(
        *(_run_track(subtask, plan.shared_state, active_connectors, overrides, config, semaphore)
          for subtask in plan.subtasks)
    ))

    await report_job_store.update(job_id, progress="Verifying findings")
    verification, verifier_id = await _verify_globally(
        plan.shared_state, active_connectors, overrides, config
    )
    assignments["verifier"] = verifier_id

    markdown = ""
    issues: list[str] = []
    for round_index in range(settings.report_max_repair_rounds + 1):
        label = "Writing report" if round_index == 0 else f"Repairing report (round {round_index})"
        await report_job_store.update(job_id, progress=label)

        writer = binding_service.select_connector(active_connectors, "writer", overrides=overrides)
        assignments["writer"] = writer.connector_id
        draft = await _write_markdown(writer, query, tracks, verification, issues, config)
        if draft is None:
            raise ValueError(f"Writer failed after {round_index + 1} attempts")

        await report_job_store.update(job_id, progress="Reviewing report")
        reviewer = binding_service.select_connector(active_connectors, "reviewer", overrides=overrides)
        assignments["reviewer"] = reviewer.connector_id
        verdict = await _review_draft(reviewer, query, draft, config)

        if verdict.approved:
            markdown = draft
            break
        issues = verdict.issues
        logger.info({
            "message": "Report draft rejected",
            "job_id": job_id,
            "round": round_index,
            "issues": len(issues),
        })

    if not markdown:
        markdown = _fallback_markdown(query, tracks, verification, issues)

    await report_job_store.update(
        job_id,
        status="completed",
        progress="Complete",
        result_markdown=markdown,
        role_assignments=assignments,
    )


async def _run_track(
    subtask,
    base_state: SharedTaskState,
    connectors: list[BaseConnector],
    overrides: dict[str, list[str]] | None,
    config: ConnectorConfig,
    semaphore: asyncio.Semaphore,
) -> TrackResult:
    async with semaphore:
        track_state = base_state.model_copy(update={
            "main_objective": subtask.objective,
            "task_context": [
                f"Subtask {subtask.subtask_id}: {subtask.title}",
                *(f"Focus: {item}" for item in subtask.focus),
            ],
        })
        focus = subtask.focus or ["Relevant facts and background"]

        research_task = ResearchTask(
            objective=subtask.objective,
            scope=focus,
            do_not_cover=["Do not produce the final answer."],
            required_output_fields=["facts", "constraints", "references", "unknowns", "confidence"],
        )
        analysis_task = AnalysisTask(
            objective=f"Analyze solution paths for: {subtask.objective}",
            scope=["Assumptions", "Tradeoffs and risks"],
            do_not_cover=["Do not produce broad background research."],
            required_output_fields=["proposed_solution", "assumptions", "tradeoffs", "risks"],
        )

        r_conn = binding_service.select_connector(connectors, "researcher", overrides=overrides)
        a_conn = binding_service.select_connector(
            connectors, "analyzer",
            excluded_ids={r_conn.connector_id} if len(connectors) > 1 else None,
            overrides=overrides,
        )

        research_out, analysis_out = await asyncio.gather(
            run_research_task(r_conn, track_state, research_task, config),
            run_analysis_task(a_conn, track_state, analysis_task, config),
            return_exceptions=True,
        )

        return TrackResult(
            subtask_id=subtask.subtask_id,
            title=subtask.title,
            research=None if isinstance(research_out, BaseException) else research_out.result,
            analysis=None if isinstance(analysis_out, BaseException) else analysis_out.result,
            error="; ".join(
                str(e) for e in (research_out, analysis_out) if isinstance(e, BaseException)
            ) or None,
        )


async def _verify_globally(
    shared_state: SharedTaskState,
    connectors: list[BaseConnector],
    overrides: dict[str, list[str]] | None,
    config: ConnectorConfig,
) -> tuple[VerificationSummary, str]:
    connector = binding_service.select_connector(connectors, "verifier", overrides=overrides)
    task = VerificationTask(
        objective=(
            "Pressure-test the combined findings of all report tracks for "
            "critical risks, hidden assumptions, and edge cases."
        ),
        scope=["Cross-track risks", "Hidden assumptions", "Edge cases"],
        do_not_cover=["Do not rewrite the report."],
        required_output_fields=["critical_risks", "hidden_assumptions", "edge_cases"],
    )
    try:
        outcome = await run_verification_task(connector, shared_state, task, config)
        result = outcome.result
        summary = VerificationSummary(
            critical_risks=result.critical_risks,
            hidden_assumptions=result.hidden_assumptions,
            edge_cases=result.edge_cases,
        )
        return summary, connector.connector_id
    except Exception as exc:
        logger.warning({"message": "Global verification failed, continuing without it", "error": str(exc)})
        return VerificationSummary(), connector.connector_id


def _tracks_payload(tracks: list[TrackResult], verification: VerificationSummary, issues: list[str]) -> str:
    sections = []
    for track in tracks:
        sections.append(json.dumps(track.model_dump(exclude_none=True), indent=2))
    payload = {
        "tracks": sections,
        "verification_summary": verification.model_dump(),
    }
    if issues:
        payload["review_issues_to_fix"] = issues
    return json.dumps(payload, indent=2)


async def _write_markdown(
    writer: BaseConnector,
    query: str,
    tracks: list[TrackResult],
    verification: VerificationSummary,
    issues: list[str],
    config: ConnectorConfig,
) -> str | None:
    prompt = _load_prompt(WRITER_PROMPT_PATH, FALLBACK_WRITER_PROMPT)
    response = await writer.query(
        prompt=prompt,
        sub_query=(
            f"Original request: {query}\n\n"
            f"Track outputs:\n{_tracks_payload(tracks, verification, issues)}"
        ),
        config=ConnectorConfig(
            timeout_s=config.timeout_s,
            max_tokens=config.max_tokens,
            temperature=0.4,
        ),
    )
    if response.status != ConnectorStatus.SUCCESS or not response.content:
        return None
    return response.content


async def _review_draft(
    reviewer: BaseConnector,
    query: str,
    draft: str,
    config: ConnectorConfig,
) -> ReviewVerdict:
    prompt = _load_prompt(REVIEWER_PROMPT_PATH, FALLBACK_REVIEWER_PROMPT)
    try:
        response = await reviewer.query(
            prompt=prompt,
            sub_query=f"Original request: {query}\n\nDraft report:\n{draft}",
            config=ConnectorConfig(timeout_s=config.timeout_s, max_tokens=1024, temperature=0.1),
        )
        if response.status != ConnectorStatus.SUCCESS or not response.content:
            raise ValueError("reviewer non-success")
        payload = _parse_json_payload(response.content)
        return ReviewVerdict.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning({"message": "Reviewer unavailable/invalid, approving draft", "error": str(exc)})
        return ReviewVerdict(approved=True, issues=[])


def _fallback_markdown(
    query: str,
    tracks: list[TrackResult],
    verification: VerificationSummary,
    issues: list[str],
) -> str:
    lines = [
        "# Research Report",
        "",
        f"**Request:** {query}",
        "",
        "## Executive Summary",
        "",
        "The synthesis model was unavailable; this report presents the collected track outputs directly.",
        "",
    ]
    for track in tracks:
        lines.append(f"## {track.title}")
        lines.append("")
        if track.research is not None:
            lines.append("### Findings")
            lines.extend(f"- {fact}" for fact in track.research.facts)
        if track.analysis is not None:
            lines.append("")
            lines.append("### Analysis")
            lines.append(str(track.analysis.proposed_solution))
        if track.error:
            lines.append("")
            lines.append(f"_Track error:_ {track.error}")
        lines.append("")
    lines.append("## Risks & Open Questions")
    lines.extend(f"- {risk}" for risk in verification.critical_risks)
    lines.extend(f"- Open: {item}" for item in verification.edge_cases)
    if issues:
        lines.append("")
        lines.append("## Unresolved Review Issues")
        lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines)
