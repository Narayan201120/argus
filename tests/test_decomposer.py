import pytest
import json
from app.orchestration.decomposer import _is_simple_query, _parse_json_response, decompose_query
from app.connectors.base import ConnectorConfig

def test_simple_query_short():
    assert _is_simple_query("What is Python?") == True

def test_simple_query_long():
    long_query = "Explain" + " ".join(["word"] * 60)
    assert _is_simple_query(long_query) == False

def test_simple_query_multiple_questions():
    assert _is_simple_query("What is X? How does Y work?") == False

def test_parse_json_valid():
    result = _parse_json_response('{"gemini": "research this"}')
    assert result["gemini"] == "research this"

def test_parse_json_with_fences():
    result = _parse_json_response('```json\n{"gemini": "test"}\n```')
    assert result["gemini"] == "test"

def test_parse_json_invalid():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response("not json at all")

@pytest.mark.asyncio
async def test_decompose_short_circuit(mock_connector):
    result = await decompose_query("Hi", [mock_connector])
    assert result == {"mock": "Hi"}

@pytest.mark.asyncio
async def test_decompose_single_connector(mock_connector):
    result = await decompose_query("Explain async in Python", [mock_connector])
    assert result == {"mock": "Explain async in Python"}