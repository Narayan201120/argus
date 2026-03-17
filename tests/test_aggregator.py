from app.connectors.base import ConnectorResponse, ConnectorStatus, TokenUsage
from app.orchestration.aggregator import (
    _build_reconciliation_summary,
    _build_synthesis_prompt,
    _labeled_concat_fallback,
)


def _response(content: str, task: str) -> ConnectorResponse:
    return ConnectorResponse(
        model_id="mock-model",
        content=content,
        latency_ms=10,
        token_usage=TokenUsage(),
        status=ConnectorStatus.SUCCESS,
        sub_query=task,
    )


def test_synthesis_prompt_includes_role_precedence():
    prompt = _build_synthesis_prompt(
        original_query="Design the system",
        response_bundle={
            "researcher": _response('{"facts":["f1"]}', "collect facts"),
            "analyzer": _response('{"proposed_solution":"x"}', "design solution"),
            "verifier": _response('{"critical_risks":["r1"]}', "pressure-test"),
        },
        system_prompt="Base synthesis prompt",
    )
    assert "Researcher owns facts" in prompt
    assert "[RESEARCHER]" in prompt
    assert "[ANALYZER]" in prompt
    assert "[VERIFIER]" in prompt
    assert "Deterministic reconciliation summary" in prompt
    assert '"facts": [' in prompt


def test_labeled_concat_fallback_orders_role_outputs():
    content = _labeled_concat_fallback(
        {
            "analyzer": _response('{"proposed_solution":"x"}', "design solution"),
            "verifier": _response('{"critical_risks":["r1"]}', "pressure-test"),
            "researcher": _response('{"facts":["f1"]}', "collect facts"),
        }
    )
    assert content.index("**RESEARCHER:**") < content.index("**ANALYZER:**")
    assert content.index("**ANALYZER:**") < content.index("**VERIFIER:**")
    assert "**RECONCILIATION SUMMARY:**" in content
    assert '"critical_risks": [' in content


def test_reconciliation_summary_flags_unsupported_assumptions():
    summary = _build_reconciliation_summary(
        {
            "researcher": _response(
                '{"facts":["Public API availability is unknown"],"constraints":["Respect rate limits"],"references":[],"unknowns":["API access"],"confidence":"medium"}',
                "collect facts",
            ),
            "analyzer": _response(
                '{"proposed_solution":"Use the API","assumptions":["API access is available"],"tradeoffs":[],"risks":["rate limit pressure"],"validation_checks":[]}',
                "design solution",
            ),
            "verifier": _response(
                '{"critical_risks":["API access may not exist"],"hidden_assumptions":["API access is available"],"edge_cases":["No API present"],"validation_requirements":["Confirm integration access"],"confidence":"high"}',
                "pressure-test",
            ),
        }
    )
    assert "API access is available" in summary["unsupported_assumptions"]
    assert summary["confidence"] in {"medium", "low"}
