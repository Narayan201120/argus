"""P5-1 report normalization + board summarization (DEC-054).

Exact Markdown matching is forbidden anywhere in the eval pipeline: reports
are compared only through normalized text or aggregate features, never
string equality.
"""

from __future__ import annotations

import re
from typing import Any

from app.eval.validators import validate_honest_gap

# Upper bound on normalized report text; keeps diff inputs small.
NORMALIZE_TRUNCATE_CHARS = 4000

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WS_RUN = re.compile(r"\s+")


def normalize_report(markdown: str) -> str:
    """Lowercase, strip punctuation/whitespace runs, truncate.

    Pure string shaping only: callers must never compare two reports with
    ``==`` on raw (or normalized) text to decide pass/fail.
    """
    lowered = markdown.lower()
    scrubbed = _NON_ALNUM.sub("", lowered)
    collapsed = _WS_RUN.sub(" ", scrubbed).strip()
    return collapsed[:NORMALIZE_TRUNCATE_CHARS]


def _status_of(claim: dict[str, Any]) -> str:
    status = claim.get("status")
    value = getattr(status, "value", status)
    return str(value).lower() if value is not None else ""


def board_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce a live-run JSONL row to verdict-relevant features.

    Expected row keys (see scripts/eval_live.py run_one): ``status``,
    ``evidence_ids``, ``claims`` (each with ``evidence_ids`` + ``status``),
    ``syntheses`` (each with ``final``). Missing keys degrade to empty.

    DEC-054: ``refused_correctly`` is an honest-answer-shape heuristic, never
    a constraint on terminal state. It is true when the board shows zero
    invented references and either carries no claims at all or only
    ``proposed`` claims that cite nothing (an honest draft, not support).
    Grounding errors on such draft claims still fail row validation
    separately; this flag only labels the answer shape.
    """
    claims_raw = row.get("claims")
    claims: list[dict[str, Any]] = [c for c in claims_raw if isinstance(c, dict)] if isinstance(
        claims_raw, list
    ) else []
    evidence_raw = row.get("evidence_ids")
    evidence_ids: set[str] = set(e for e in evidence_raw if isinstance(e, str)) if isinstance(
        evidence_raw, list
    ) else set()
    syntheses_raw = row.get("syntheses")
    syntheses: list[Any] = list(syntheses_raw) if isinstance(syntheses_raw, list) else []

    grounded = 0
    for claim in claims:
        cited = claim.get("evidence_ids")
        if (
            isinstance(cited, list)
            and bool(cited)
            and all(isinstance(v, str) and v in evidence_ids for v in cited)
        ):
            grounded += 1
    grounded_ratio = (grounded / len(claims)) if claims else 1.0

    has_final_synthesis = any(
        isinstance(s, dict) and s.get("final") is True for s in syntheses
    )

    board = {
        "evidence": [{"id": eid} for eid in sorted(evidence_ids)],
        "claims": claims,
    }
    # Invented source_refs anywhere (claim statements, syntheses) are
    # fabrication in any status, so they veto the refusal label. Empty
    # citations on proposed claims do not: they are honest drafts.
    honest_gap_errors = validate_honest_gap(board, syntheses)
    zero_invented_refs = not honest_gap_errors

    if not claims:
        refused_correctly = zero_invented_refs
    else:
        refused_correctly = zero_invented_refs and all(
            _status_of(claim) == "proposed" and not claim.get("evidence_ids") for claim in claims
        )

    return {
        "terminal": row.get("status"),
        "claim_count": len(claims),
        "grounded_ratio": grounded_ratio,
        "has_final_synthesis": has_final_synthesis,
        "refused_correctly": refused_correctly,
    }
