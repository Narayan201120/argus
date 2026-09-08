"""P5-0 mock-only eval package (DEC-054)."""

from app.eval.corpus import EvalCase, EvalExpectation, load_corpus
from app.eval.validators import (
    validate_board_shape,
    validate_claim_grounding,
    validate_honest_gap,
)

__all__ = [
    "EvalCase",
    "EvalExpectation",
    "load_corpus",
    "validate_board_shape",
    "validate_claim_grounding",
    "validate_honest_gap",
]
