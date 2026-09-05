"""P4-2 evidence board renderer (DEC-053, backend only)."""

from app.evidence.models import Investigation

BOARD_MAX_EVIDENCE = 20
BOARD_EVIDENCE_CHARS = 1000
BOARD_MAX_CHARS = 12000
BOARD_MAX_CLAIMS = 20


def format_board(inv: Investigation) -> str:
    """Render an investigation board as deterministic plain text for worker prompts.

    Board order is preserved (evidence/claims are truncated, never re-sorted).
    Never returns an empty string: an empty board renders a valid
    "no evidence yet" prompt instead.
    """
    evidence = inv.board.evidence[:BOARD_MAX_EVIDENCE]
    claims = inv.board.claims[:BOARD_MAX_CLAIMS]

    lines = [
        f"Investigation {inv.id}",
        f"Query: {inv.query}",
        (
            f"Evidence: {len(inv.board.evidence)} "
            f"(showing {len(evidence)}) | "
            f"Claims: {len(inv.board.claims)} (showing {len(claims)})"
        ),
        "",
    ]

    if not evidence and not claims:
        lines.append("Status: no evidence yet — gather evidence before analysis.")
        lines.append("Next: run radar/rag tools to collect evidence for the query above.")
        return "\n".join(lines)[:BOARD_MAX_CHARS] or "no evidence yet"

    lines.append("Evidence:")
    for index, item in enumerate(evidence, start=1):
        content = item.content[:BOARD_EVIDENCE_CHARS]
        lines.append(
            f"{index}. [id={item.id}] source={item.source_ref} "
            f"type={item.type} confidence={item.confidence:.2f}"
        )
        lines.append(f"   {content}")

    lines.append("")
    lines.append("Claims:")
    for index, claim in enumerate(claims, start=1):
        evidence_ids = ", ".join(claim.evidence_ids)
        lines.append(
            f"{index}. [id={claim.id}] confidence={claim.confidence:.2f} "
            f"status={claim.status.value} evidence_ids=[{evidence_ids}]"
        )
        lines.append(f"   {claim.statement}")

    rendered = "\n".join(lines)
    return rendered[:BOARD_MAX_CHARS] or "no evidence yet"
