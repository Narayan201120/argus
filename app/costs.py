"""P4-5b investigation cost budget (DEC-053, backend only, mock-only).

All values here are estimates for budget gating, not real billing.
Real spend lives in provider consoles (LLM) and Tavily (web).
Cap <= 0 disables the cost check; see InvestigationManager.check_budgets.
"""

from app.config import settings


def estimate_llm_cost(connector_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD for one LLM call from per-1k token tables.

    Unknown connector_id falls back to the "default" table entry.
    Negative token counts clamp to zero. Pure function.
    """
    prompt_n = max(int(prompt_tokens), 0)
    completion_n = max(int(completion_tokens), 0)
    in_table = settings.cost_usd_per_1k_input_tokens
    out_table = settings.cost_usd_per_1k_output_tokens
    in_rate = in_table.get(connector_id, in_table.get("default", 0.0))
    out_rate = out_table.get(connector_id, out_table.get("default", 0.0))
    return prompt_n / 1000.0 * float(in_rate) + completion_n / 1000.0 * float(out_rate)


def web_search_cost() -> float:
    """Estimated USD per reserved web_search call. Pure function."""
    return float(settings.cost_usd_per_web_search)


def web_fetch_cost() -> float:
    """Estimated USD per reserved web_fetch call. Pure function."""
    return float(settings.cost_usd_per_web_fetch)
