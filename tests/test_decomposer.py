from app.orchestration.decomposer import _is_simple_query


def test_simple_query_short():
    assert _is_simple_query("What is Python?") is True

def test_simple_query_long():
    long_query = "Explain" + " ".join(["word"] * 60)
    assert _is_simple_query(long_query) is False

def test_simple_query_multiple_questions():
    assert _is_simple_query("What is X? How does Y work?") is False
