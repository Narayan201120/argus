"""P5-0 pure board validators (DEC-054, mock-only).

Every function returns an error list (empty means pass) so runners can print
diagnostics. Claim grounding uses ID-SET membership only: a claim is grounded
when each of its evidence_ids names an evidence item present on the board.
String/substring matching is never used.

DEC-054: verdicts never constrain terminal state. A refused-correctly board
(empty, zero claims, zero citations) passes all validators in any terminal
state; only fabrication fails.
"""

from __future__ import annotations

import re
from typing import Any

from app.evidence.models import Board

EVIDENCE_REQUIRED = ("id", "investigation_id", "source_ref", "content", "type", "confidence", "created_at")
CLAIM_REQUIRED = ("id", "investigation_id", "statement", "confidence", "evidence_ids")
SOURCE_FRAGMENT = re.compile(r"source=([^\s,\]]+)")
SYNTHESIS_REF_KEYS = ("source_refs", "sources", "citations")


def _board_items(board: Board | dict[str, Any]) -> tuple[Any, Any]:
    if isinstance(board, Board):
        return board.evidence, board.claims
    if isinstance(board, dict):
        return board.get("evidence"), board.get("claims")
    return None, None


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        dumped: Any = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def validate_board_shape(board_dict: Board | dict[str, Any]) -> list[str]:
    """Check evidence/claim required fields and confidence ranges."""
    errors: list[str] = []
    evidence, claims = _board_items(board_dict)
    if not isinstance(evidence, list):
        errors.append("board.evidence is missing or not a list")
        evidence = []
    if not isinstance(claims, list):
        errors.append("board.claims is missing or not a list")
        claims = []
    seen_evidence: set[str] = set()
    for index, raw in enumerate(evidence):
        item = _as_dict(raw)
        where = f"evidence[{index}]"
        for field in EVIDENCE_REQUIRED:
            if field not in item:
                errors.append(f"{where} is missing required field {field!r}")
        for field in ("id", "investigation_id", "source_ref", "content", "type"):
            value = item.get(field)
            if field in item and (not isinstance(value, str) or not value):
                errors.append(f"{where}.{field} must be a non-empty string")
        confidence = item.get("confidence")
        if "confidence" in item and (
            not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            errors.append(f"{where}.confidence={confidence!r} is not in [0.0, 1.0]")
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            if item_id in seen_evidence:
                errors.append(f"{where}.id={item_id!r} is duplicated")
            seen_evidence.add(item_id)
    seen_claims: set[str] = set()
    for index, raw in enumerate(claims):
        item = _as_dict(raw)
        where = f"claims[{index}]"
        for field in CLAIM_REQUIRED:
            if field not in item:
                errors.append(f"{where} is missing required field {field!r}")
        for field in ("id", "investigation_id", "statement"):
            value = item.get(field)
            if field in item and (not isinstance(value, str) or not value):
                errors.append(f"{where}.{field} must be a non-empty string")
        confidence = item.get("confidence")
        if "confidence" in item and (
            not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            errors.append(f"{where}.confidence={confidence!r} is not in [0.0, 1.0]")
        evidence_ids = item.get("evidence_ids")
        if "evidence_ids" in item and (
            not isinstance(evidence_ids, list) or not all(isinstance(v, str) for v in evidence_ids)
        ):
            errors.append(f"{where}.evidence_ids must be a list of strings")
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            if item_id in seen_claims:
                errors.append(f"{where}.id={item_id!r} is duplicated")
            seen_claims.add(item_id)
    return errors


def validate_claim_grounding(board: Board | dict[str, Any]) -> tuple[float, list[str]]:
    """Check every claim evidence_id names an evidence id on the board.

    Returns (grounded_ratio, errors). A claim is grounded when it cites at
    least one evidence id and every cited id exists on the board. A board
    with no claims scores 1.0 (refused-correctly: zero fabrication passes).
    """
    _, claims = _board_items(board)
    evidence, _ = _board_items(board)
    evidence_ids: set[str] = set()
    if isinstance(evidence, list):
        for raw in evidence:
            item_id = _as_dict(raw).get("id")
            if isinstance(item_id, str) and item_id:
                evidence_ids.add(item_id)
    if not isinstance(claims, list):
        return 0.0, ["board.claims is missing or not a list"]
    if not claims:
        return 1.0, []
    errors: list[str] = []
    grounded = 0
    for raw in claims:
        item = _as_dict(raw)
        claim_id = item.get("id", "?")
        cited = item.get("evidence_ids")
        if not isinstance(cited, list) or not cited:
            errors.append(f"claim {claim_id!r} cites no evidence (ungrounded)")
            continue
        dangling = [v for v in cited if v not in evidence_ids]
        if dangling:
            errors.append(f"claim {claim_id!r} cites missing evidence ids {dangling}")
            continue
        grounded += 1
    return grounded / len(claims), errors


def _cited_refs(synthesis: Any) -> list[str]:
    """Extract source_refs a synthesis entry claims to rely on."""
    refs: list[str] = []
    if isinstance(synthesis, dict):
        for key in SYNTHESIS_REF_KEYS:
            values = synthesis.get(key)
            if isinstance(values, list):
                refs.extend(v for v in values if isinstance(v, str))
        for key in ("markdown", "text", "content"):
            body = synthesis.get(key)
            if isinstance(body, str):
                refs.extend(SOURCE_FRAGMENT.findall(body))
    elif isinstance(synthesis, str):
        refs.extend(SOURCE_FRAGMENT.findall(synthesis))
    return refs


def _status_value(item: dict[str, Any]) -> str:
    status = item.get("status")
    value = getattr(status, "value", status)
    return str(value) if value is not None else ""


def validate_honest_gap(
    board: Board | dict[str, Any], syntheses: list[Any] | None = None
) -> list[str]:
    """Flag invented source_refs and unsupported claims posed as supported.

    Invented means cited anywhere in claims or syntheses but absent from the
    board's evidence source_refs. A claim with status supported that cites no
    evidence (or cites missing evidence) is fabrication, not support.
    """
    errors: list[str] = []
    evidence, claims = _board_items(board)
    board_refs: set[str] = set()
    if isinstance(evidence, list):
        for raw in evidence:
            ref = _as_dict(raw).get("source_ref")
            if isinstance(ref, str) and ref:
                board_refs.add(ref)
    evidence_ids: set[str] = set()
    if isinstance(evidence, list):
        for raw in evidence:
            item_id = _as_dict(raw).get("id")
            if isinstance(item_id, str) and item_id:
                evidence_ids.add(item_id)
    claim_list = claims if isinstance(claims, list) else []
    for raw in claim_list:
        item = _as_dict(raw)
        claim_id = item.get("id", "?")
        cited = item.get("evidence_ids")
        cited_ok = isinstance(cited, list) and bool(cited) and all(v in evidence_ids for v in cited)
        if _status_value(item) == "supported" and not cited_ok:
            errors.append(f"claim {claim_id!r} is marked supported without grounded evidence")
        statement = item.get("statement")
        if isinstance(statement, str):
            for ref in SOURCE_FRAGMENT.findall(statement):
                if ref not in board_refs:
                    errors.append(f"claim {claim_id!r} cites invented source_ref {ref!r}")
    for index, synthesis in enumerate(syntheses or []):
        for ref in _cited_refs(synthesis):
            if ref not in board_refs:
                errors.append(f"synthesis[{index}] cites invented source_ref {ref!r}")
    return errors
