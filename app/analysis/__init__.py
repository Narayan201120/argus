"""P4-2 analysis public surface (DEC-053, backend only)."""

from app.analysis.board import (
    BOARD_EVIDENCE_CHARS,
    BOARD_MAX_CHARS,
    BOARD_MAX_CLAIMS,
    BOARD_MAX_EVIDENCE,
    format_board,
)
from app.analysis.workers import (
    ANALYSIS_PROMPT,
    CRITIQUE_PROMPT,
    GAP_PROMPT,
    WORKER_ANALYSIS,
    WORKER_CRITIQUE,
    WORKER_GAP,
    AnalysisOutput,
    Challenge,
    ClaimDraft,
    CritiqueOutput,
    GapOutput,
    WorkerError,
    analyze_board,
    assess_gaps,
    critique_board,
)

__all__ = [
    "ANALYSIS_PROMPT",
    "CRITIQUE_PROMPT",
    "GAP_PROMPT",
    "WORKER_ANALYSIS",
    "WORKER_CRITIQUE",
    "WORKER_GAP",
    "AnalysisOutput",
    "BOARD_EVIDENCE_CHARS",
    "BOARD_MAX_CHARS",
    "BOARD_MAX_CLAIMS",
    "BOARD_MAX_EVIDENCE",
    "Challenge",
    "ClaimDraft",
    "CritiqueOutput",
    "GapOutput",
    "WorkerError",
    "analyze_board",
    "assess_gaps",
    "critique_board",
    "format_board",
]
