import json
from pathlib import Path

from pydantic import ValidationError

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
)
from app.orchestration.contracts import AnalysisResult, ResearchResult, VerificationResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYNTHESIS_PROMPT_PATH = Path("prompts/synthesis_v1.txt")

FALLBACK_PROMPT = (
    "You are a synthesis AI. You received the original user query and labeled responses "
    "from multiple AI models, each addressing a different aspect of the query. "
    "Synthesize these into one coherent, non-redundant, authoritative answer."
)

ROLE_PRECEDENCE_INSTRUCTIONS = """Role precedence rules:
- Researcher owns facts, constraints, references, and unknowns.
- Analyzer owns proposed solution logic, assumptions, tradeoffs, risks, and validation checks.
- Verifier owns critical risks, hidden assumptions, edge cases, and validation requirements.
- Do not silently smooth over conflicts between roles.
- If the analyzer assumes something not supported by the researcher, label it as an assumption rather than a fact.
- If the verifier identifies a material risk or hidden assumption, surface it clearly in the final answer.
- Prefer explicit uncertainty over unsupported certainty.
"""


def _load_synthesis_prompt() -> str:
    try:
        return SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning({"message": "synthesis_v1.txt not found, using inline fallback"})
        return FALLBACK_PROMPT


def _role_label(connector_id: str) -> str:
    return {
        "researcher": "RESEARCHER",
        "analyzer": "ANALYZER",
        "verifier": "VERIFIER",
    }.get(connector_id, connector_id.upper())


def _parse_json_content(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _parse_role_outputs(response_bundle: dict[str, ConnectorResponse]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    role_models = {
        "researcher": ResearchResult,
        "analyzer": AnalysisResult,
        "verifier": VerificationResult,
    }

    for role, model in role_models.items():
        response = response_bundle.get(role)
        if not response or response.status != ConnectorStatus.SUCCESS or not response.content:
            continue
        try:
            parsed[role] = model.model_validate(_parse_json_content(response.content))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning({
                "message": "Failed to parse role output for reconciliation",
                "role": role,
                "error": str(exc),
            })
    return parsed


def _build_clean_role_sections(
    response_bundle: dict[str, ConnectorResponse],
    parsed_roles: dict[str, object],
) -> list[str]:
    ordered_roles = ["researcher", "analyzer", "verifier"]
    sections: list[str] = []

    for connector_id in ordered_roles:
        parsed = parsed_roles.get(connector_id)
        response = response_bundle.get(connector_id)
        if parsed is not None and response is not None:
            task_label = response.sub_query or "general"
            cleaned_payload = json.dumps(parsed.model_dump(exclude_none=True), indent=2)
            sections.append(
                f"--- [{_role_label(connector_id)}] (Task: {task_label}) ---\n{cleaned_payload}"
            )

    for connector_id, response in response_bundle.items():
        if connector_id in ordered_roles:
            continue
        if response.status == ConnectorStatus.SUCCESS and response.content:
            task_label = response.sub_query or "general"
            sections.append(
                f"--- [{_role_label(connector_id)}] (Task: {task_label}) ---\n{response.content}"
            )

    return sections


def _normalize_text_list(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _find_unsupported_assumptions(
    research: ResearchResult | None,
    analysis: AnalysisResult | None,
    verification: VerificationResult | None,
) -> list[str]:
    if analysis is None:
        return []

    research_text = " ".join(
        (research.facts + research.constraints + research.references + research.unknowns)
        if research is not None else []
    ).casefold()
    verifier_text = " ".join(
        (verification.hidden_assumptions + verification.critical_risks + verification.edge_cases)
        if verification is not None else []
    ).casefold()

    unsupported = []
    for assumption in analysis.assumptions:
        tokenized = [token.casefold() for token in assumption.replace("/", " ").replace("-", " ").split() if len(token) > 3]
        supported = any(token in research_text for token in tokenized) if research_text else False
        challenged = any(token in verifier_text for token in tokenized) if verifier_text else False
        if not supported or challenged:
            unsupported.append(assumption)
    return _normalize_text_list(unsupported)


def _find_missing_validation_coverage(
    analysis: AnalysisResult | None,
    verification: VerificationResult | None,
) -> list[str]:
    if analysis is None or verification is None:
        return []

    analysis_checks_text = " ".join(analysis.validation_checks).casefold()
    missing = []
    for requirement in verification.validation_requirements:
        tokens = [token.casefold() for token in requirement.replace("/", " ").replace("-", " ").split() if len(token) > 3]
        covered = any(token in analysis_checks_text for token in tokens) if analysis_checks_text else False
        if not covered:
            missing.append(requirement)
    return _normalize_text_list(missing)


def _find_constraint_risk_conflicts(
    research: ResearchResult | None,
    analysis: AnalysisResult | None,
    verification: VerificationResult | None,
) -> list[str]:
    if research is None:
        return []

    risk_text = " ".join(
        (analysis.risks if analysis is not None else [])
        + (verification.critical_risks if verification is not None else [])
    ).casefold()

    conflicts = []
    for constraint in research.constraints:
        tokens = [token.casefold() for token in constraint.replace("/", " ").replace("-", " ").split() if len(token) > 3]
        if any(token in risk_text for token in tokens):
            conflicts.append(f"Research constraint challenged by risk: {constraint}")
    return _normalize_text_list(conflicts)


def _build_reconciliation_summary(response_bundle: dict[str, ConnectorResponse]) -> dict[str, object]:
    parsed = _parse_role_outputs(response_bundle)
    research = parsed.get("researcher") if isinstance(parsed.get("researcher"), ResearchResult) else None
    analysis = parsed.get("analyzer") if isinstance(parsed.get("analyzer"), AnalysisResult) else None
    verification = parsed.get("verifier") if isinstance(parsed.get("verifier"), VerificationResult) else None

    unsupported_assumptions = _find_unsupported_assumptions(research, analysis, verification)
    missing_validation_coverage = _find_missing_validation_coverage(analysis, verification)
    constraint_risk_conflicts = _find_constraint_risk_conflicts(research, analysis, verification)
    material_risks = _normalize_text_list(
        (analysis.risks if analysis is not None else [])
        + (verification.critical_risks if verification is not None else [])
    )
    missing_evidence = _normalize_text_list(research.unknowns if research is not None else [])
    conflicts = _normalize_text_list(
        [f"Unsupported analyzer assumption: {item}" for item in unsupported_assumptions]
        + [f"Verifier surfaced hidden assumption: {item}" for item in (verification.hidden_assumptions if verification is not None else [])]
        + [f"Missing validation coverage: {item}" for item in missing_validation_coverage]
        + constraint_risk_conflicts
    )

    confidence = "high"
    if conflicts or material_risks or missing_validation_coverage:
        confidence = "medium"
    if (
        len(conflicts) >= 2
        or len(missing_evidence) >= 2
        or len(missing_validation_coverage) >= 2
    ):
        confidence = "low"

    return {
        "unsupported_assumptions": unsupported_assumptions,
        "material_risks": material_risks,
        "missing_evidence": missing_evidence,
        "missing_validation_coverage": missing_validation_coverage,
        "constraint_risk_conflicts": constraint_risk_conflicts,
        "conflicts": conflicts,
        "confidence": confidence,
        "parsed_roles": sorted(parsed.keys()),
    }


def _build_synthesis_prompt(
    original_query: str,
    response_bundle: dict[str, ConnectorResponse],
    system_prompt: str,
) -> str:
    parsed_roles = _parse_role_outputs(response_bundle)
    sections = _build_clean_role_sections(response_bundle, parsed_roles)
    reconciliation_summary = _build_reconciliation_summary(response_bundle)
    assembled = "\n\n".join(sections)
    return (
        f"{system_prompt}\n\n"
        f"{ROLE_PRECEDENCE_INSTRUCTIONS}\n"
        f"Original user query: {original_query}\n\n"
        f"Role outputs:\n{assembled}\n\n"
        f"Deterministic reconciliation summary:\n{json.dumps(reconciliation_summary, indent=2)}\n\n"
        "Produce a final answer that is grounded in the role outputs and the reconciliation summary above. "
        "Call out important assumptions, conflicts, or risks when they materially affect confidence."
    )


def _labeled_concat_fallback(
    response_bundle: dict[str, ConnectorResponse],
) -> str:
    lines = [
        "The following role-scoped outputs were collected by ARGUS:\n"
    ]
    parsed_roles = _parse_role_outputs(response_bundle)
    for connector_id in ["researcher", "analyzer", "verifier"]:
        parsed = parsed_roles.get(connector_id)
        if parsed is not None:
            lines.append(f"**{_role_label(connector_id)}:**\n{json.dumps(parsed.model_dump(exclude_none=True), indent=2)}\n")

    reconciliation_summary = _build_reconciliation_summary(response_bundle)
    lines.append(f"**RECONCILIATION SUMMARY:**\n{json.dumps(reconciliation_summary, indent=2)}\n")

    for connector_id, response in response_bundle.items():
        if connector_id in {"researcher", "analyzer", "verifier"}:
            continue
        if response.status == ConnectorStatus.SUCCESS and response.content:
            lines.append(f"**{_role_label(connector_id)}:**\n{response.content}\n")

    if len(lines) == 2 and "No successful" not in lines[0]:
        return "No successful responses were collected from any connector."
    return "\n".join(lines)


async def synthesize(
    original_query: str,
    response_bundle: dict[str, ConnectorResponse],
    synthesizer_chain: list[BaseConnector],
    config: ConnectorConfig | None = None,
) -> tuple[str, str, ConnectorResponse | None]:
    if config is None:
        config = ConnectorConfig(max_tokens=4096, temperature=0.3)

    successful = {
        cid: r
        for cid, r in response_bundle.items()
        if r.status == ConnectorStatus.SUCCESS and r.content
    }

    if not successful:
        logger.warning({"message": "All connectors failed - returning diagnostic response"})
        return (
            "All configured connectors failed to respond. "
            "Please check your API keys and connector availability.",
            "system",
            None,
        )

    if len(successful) == 1:
        _, r = next(iter(successful.items()))
        return r.content, r.model_id, r

    system_prompt = _load_synthesis_prompt()
    synthesis_input = _build_synthesis_prompt(original_query, successful, system_prompt)

    for synthesizer in synthesizer_chain:
        if not synthesizer.is_available:
            logger.info({
                "message": "Synthesizer unavailable, trying next",
                "synthesizer": synthesizer.connector_id,
            })
            continue

        try:
            logger.info({"message": "Attempting synthesis", "synthesizer": synthesizer.connector_id})
            response = await synthesizer.query(
                prompt="You are the synthesis layer of an AI orchestration system.",
                sub_query=synthesis_input,
                config=config,
            )

            if response.status == ConnectorStatus.SUCCESS and response.content:
                logger.info({
                    "message": "Synthesis succeeded",
                    "synthesizer": synthesizer.connector_id,
                    "latency_ms": response.latency_ms,
                })
                return response.content, synthesizer.connector_id, response

            logger.warning({
                "message": "Synthesizer responded with non-success",
                "synthesizer": synthesizer.connector_id,
                "status": response.status,
            })

        except Exception as e:
            logger.error({
                "message": "Synthesizer exception",
                "synthesizer": synthesizer.connector_id,
                "error": str(e),
            })
            continue

    logger.warning({"message": "All synthesizers exhausted - using labeled concatenation fallback"})
    content = _labeled_concat_fallback(successful)
    return content, "fallback_concat", None
