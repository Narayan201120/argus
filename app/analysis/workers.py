"""P4-2 analysis workers (DEC-053, backend only).

Each worker calls a real LLM connector once (plus one failover retry) and
parses a single JSON object from the reply. Failures surface as
:class:`WorkerError` data so the investigation loop can record them.

Worker name strings (the loop and dashboards key off these): "analysis",
"critique", "gap".
"""

import json
import time
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
)
from app.connectors.registry import registry
from app.costs import estimate_llm_cost
from app.investigations import manager
from app.metrics import WORKER_CALLS, WORKER_LATENCY, record_role_tokens

WORKER_ANALYSIS = "analysis"
WORKER_CRITIQUE = "critique"
WORKER_GAP = "gap"

ANALYSIS_PROMPT = (
    "You are the ARGUS analysis worker. Propose evidence-grounded claims. "
    'Return only one JSON object, no prose: {"claims": '
    '[{"statement": str, "confidence": 0..1, "evidence_ids": [str]}]}.'
)
CRITIQUE_PROMPT = (
    "You are the ARGUS critique worker. Challenge weak or unsupported claims. "
    'Return only one JSON object, no prose: {"challenges": '
    '[{"target_claim_id": str|null, "point": str, "severity": 0..1}]}.'
)
GAP_PROMPT = (
    "You are the ARGUS gap worker. Judge evidence sufficiency and propose "
    "follow-up tool queries. Return only one JSON object, no prose: "
    '{"sufficient": bool, "radar_query": str, "rag_query": str, "web_query": str, "rationale": str}. '
    "Provide a web follow-up query in web_query, or empty string when web search is not needed."
)

WORKER_CONFIGS: dict[str, ConnectorConfig] = {
    WORKER_ANALYSIS: ConnectorConfig(temperature=0.2, max_tokens=1500),
    WORKER_CRITIQUE: ConnectorConfig(temperature=0.2, max_tokens=1200),
    WORKER_GAP: ConnectorConfig(temperature=0.2, max_tokens=800),
}


class WorkerError(Exception):
    """Worker failure as data; the investigation loop catches this."""


class ClaimDraft(BaseModel):
    statement: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = []


class AnalysisOutput(BaseModel):
    claims: list[ClaimDraft] = []


class Challenge(BaseModel):
    target_claim_id: str | None = None
    point: str = Field(min_length=1, max_length=2000)
    severity: float = Field(ge=0.0, le=1.0)


class CritiqueOutput(BaseModel):
    challenges: list[Challenge] = []


class GapOutput(BaseModel):
    sufficient: bool
    radar_query: str = ""
    rag_query: str = ""
    web_query: str = ""
    rationale: str = ""


ResultT = TypeVar("ResultT", bound=BaseModel)


def _pick_connector() -> BaseConnector:
    """Select the connector for analysis workers."""
    pinned = settings.analysis_connector_id.strip()
    if pinned:
        connector = registry.get(pinned)
        if connector is None:
            raise WorkerError(f"provider_error: unknown analysis connector {pinned!r}")
        if not connector.is_available:
            raise WorkerError(f"provider_error: analysis connector {pinned!r} unavailable")
        return connector
    available = registry.available()
    if not available:
        raise WorkerError("provider_error: no available connectors")
    return available[0]


def _failover_candidate(exclude_id: str) -> BaseConnector | None:
    """Next available connector other than the one that just failed."""
    for connector in registry.available():
        if connector.connector_id != exclude_id:
            return connector
    return None


def _provider_error(response: ConnectorResponse) -> WorkerError:
    detail = response.error or response.status.value
    return WorkerError(f"provider_error: {detail}")


async def _call_with_failover(
    prompt: str,
    sub_query: str,
    config: ConnectorConfig,
) -> tuple[ConnectorResponse, BaseConnector]:
    """Call the picked connector, retrying once with the next available one.

    Both raised exceptions and non-success ConnectorResponse replies count
    as provider failures and trigger the single failover attempt.
    """
    primary = _pick_connector()
    try:
        response = await primary.query(prompt, sub_query, config)
    except Exception as exc:  # noqa: BLE001 - any transport/SDK error is a provider failure
        fallback = _failover_candidate(primary.connector_id)
        if fallback is None:
            raise WorkerError(f"provider_error: {exc}") from exc
        try:
            retry = await fallback.query(prompt, sub_query, config)
        except Exception as exc2:  # noqa: BLE001 - failover also failed
            raise WorkerError(f"provider_error: {exc2}") from exc2
        if retry.status != ConnectorStatus.SUCCESS or not retry.content:
            raise _provider_error(retry) from None
        return retry, fallback

    if response.status == ConnectorStatus.SUCCESS and response.content:
        return response, primary

    fallback = _failover_candidate(primary.connector_id)
    if fallback is None:
        raise _provider_error(response)
    try:
        retry = await fallback.query(prompt, sub_query, config)
    except Exception as exc:  # noqa: BLE001 - failover also failed
        raise WorkerError(f"provider_error: {exc}") from exc
    if retry.status != ConnectorStatus.SUCCESS or not retry.content:
        raise _provider_error(retry) from None
    return retry, fallback


def _extract_json_object(text: str) -> dict:
    """Extract the first {...} JSON object from response text."""
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


async def _run_worker(
    worker: str,
    system_prompt: str,
    model: type[ResultT],
    board_text: str,
    query: str,
    config: ConnectorConfig,
    *,
    investigation_id: str | None = None,
) -> ResultT:
    start = time.perf_counter()
    status = "ok"
    try:
        task = (
            f"{system_prompt}\n\n"
            f"Original investigation query: {query}\n\n"
            f"Evidence board:\n{board_text}\n\n"
            "Return only one JSON object, no prose."
        )
        response, connector = await _call_with_failover(
            prompt=query, sub_query=task, config=config
        )
        try:
            payload = _extract_json_object(response.content)
        except (ValueError, json.JSONDecodeError) as exc:
            status = "parse_error"
            raise WorkerError(f"parse_error: {exc}") from exc
        try:
            result = model.model_validate(payload)
        except ValidationError as exc:
            status = "parse_error"
            raise WorkerError(f"parse_error: {exc}") from exc
        record_role_tokens(worker, connector.connector_id, response.token_usage)
        if investigation_id and response.token_usage is not None:
            amount = estimate_llm_cost(
                connector.connector_id,
                response.token_usage.prompt_tokens,
                response.token_usage.completion_tokens,
            )
            await manager.add_cost(investigation_id, amount)
        return result
    except WorkerError as exc:
        message = str(exc)
        if message.startswith("parse_error:"):
            status = "parse_error"
        else:
            status = "provider_error"
            if not message.startswith("provider_error:"):
                raise WorkerError(f"provider_error: {message}") from exc
        raise
    except Exception as exc:  # noqa: BLE001 - unexpected failure is a provider failure
        status = "provider_error"
        raise WorkerError(f"provider_error: {exc}") from exc
    finally:
        WORKER_LATENCY.labels(worker=worker).observe(time.perf_counter() - start)
        WORKER_CALLS.labels(worker=worker, status=status).inc()


async def analyze_board(
    board_text: str, query: str, *, investigation_id: str | None = None
) -> AnalysisOutput:
    """Propose evidence-grounded claim drafts for the board."""
    return await _run_worker(
        WORKER_ANALYSIS, ANALYSIS_PROMPT, AnalysisOutput,
        board_text, query, WORKER_CONFIGS[WORKER_ANALYSIS],
        investigation_id=investigation_id,
    )


async def critique_board(
    board_text: str, query: str, *, investigation_id: str | None = None
) -> CritiqueOutput:
    """Challenge weak or unsupported claims on the board."""
    return await _run_worker(
        WORKER_CRITIQUE, CRITIQUE_PROMPT, CritiqueOutput,
        board_text, query, WORKER_CONFIGS[WORKER_CRITIQUE],
        investigation_id=investigation_id,
    )


async def assess_gaps(
    board_text: str, query: str, *, investigation_id: str | None = None
) -> GapOutput:
    """Judge evidence sufficiency and propose follow-up tool queries."""
    return await _run_worker(
        WORKER_GAP, GAP_PROMPT, GapOutput,
        board_text, query, WORKER_CONFIGS[WORKER_GAP],
        investigation_id=investigation_id,
    )
