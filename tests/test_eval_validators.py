"""P5-0 validator unit tests (DEC-054, mock-only, no network)."""

from typing import Any

from app.eval.validators import (
    validate_board_shape,
    validate_claim_grounding,
    validate_honest_gap,
)


def _evidence(item_id: str, ref: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "investigation_id": "inv-1",
        "source_ref": ref,
        "content": "finding content",
        "type": "text",
        "confidence": 0.8,
        "created_at": 1700000000.0,
    }


def _claim(item_id: str, evidence_ids: list[str], status: str = "supported") -> dict[str, Any]:
    return {
        "id": item_id,
        "investigation_id": "inv-1",
        "statement": "mock statement",
        "confidence": 0.7,
        "evidence_ids": evidence_ids,
        "status": status,
    }


def _grounded_board() -> dict[str, Any]:
    return {
        "evidence": [_evidence("ev-1", "ref-a"), _evidence("ev-2", "ref-b")],
        "claims": [_claim("cl-1", ["ev-1", "ev-2"])],
    }


def test_shape_passes_on_grounded_board() -> None:
    assert validate_board_shape(_grounded_board()) == []


def test_shape_flags_missing_fields_and_bad_confidence() -> None:
    board = {"evidence": [{"id": "ev-1", "confidence": 1.5}], "claims": [{"id": "cl-1"}]}
    errors = validate_board_shape(board)
    assert any("missing required field" in e for e in errors)
    assert any("confidence=1.5" in e for e in errors)


def test_shape_flags_duplicate_ids() -> None:
    board = {
        "evidence": [_evidence("ev-1", "ref-a"), _evidence("ev-1", "ref-b")],
        "claims": [_claim("cl-1", ["ev-1"]), _claim("cl-1", ["ev-1"])],
    }
    errors = validate_board_shape(board)
    assert any("duplicated" in e for e in errors)


def test_grounding_passes_on_id_set_membership() -> None:
    ratio, errors = validate_claim_grounding(_grounded_board())
    assert errors == []
    assert ratio == 1.0


def test_grounding_flags_dangling_and_empty_citations() -> None:
    board = {
        "evidence": [_evidence("ev-1", "ref-a")],
        "claims": [_claim("cl-1", ["ev-1"]), _claim("cl-2", ["ev-missing"]), _claim("cl-3", [])],
    }
    ratio, errors = validate_claim_grounding(board)
    assert len(errors) == 2
    assert any("ev-missing" in e for e in errors)
    assert any("cl-3" in e for e in errors)
    assert ratio == 1 / 3


def test_grounding_empty_board_scores_vacuous_pass() -> None:
    ratio, errors = validate_claim_grounding({"evidence": [], "claims": []})
    assert errors == []
    assert ratio == 1.0


def test_honest_gap_flags_invented_synthesis_ref() -> None:
    board = _grounded_board()
    syntheses = [{"markdown": "says source=ref-a and source=ref-ghost", "source_refs": ["ref-ghost"]}]
    errors = validate_honest_gap(board, syntheses)
    assert any("ref-ghost" in e for e in errors)
    assert not any("ref-a" in e for e in errors)


def test_honest_gap_flags_supported_without_evidence() -> None:
    board = {"evidence": [_evidence("ev-1", "ref-a")], "claims": [_claim("cl-1", [])]}
    errors = validate_honest_gap(board, [])
    assert any("supported without grounded evidence" in e for e in errors)


def test_honest_gap_refused_correctly_passes() -> None:
    board: dict[str, Any] = {"evidence": [], "claims": []}
    assert validate_board_shape(board) == []
    assert validate_honest_gap(board, [{"markdown": "I cannot answer.", "source_refs": []}]) == []
