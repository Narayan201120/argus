"""Stage P2-1 - embedding backends (mock-only, no live calls)."""

import pytest

from app.config import settings
from app.embeddings import GeminiEmbedder, OpenAIEmbedder, cosine_similarity, get_embedder


def test_cosine_similarity_basic_directions():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0
    assert abs(cosine_similarity([1, 1], [1, 0]) - 0.7071067811865476) < 1e-9


def test_cosine_similarity_handles_degenerate_inputs():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 2], [1]) == 0.0  # length mismatch


@pytest.mark.parametrize(
    ("provider", "gemini_key", "openai_key", "expected"),
    [
        ("none", "g", "o", None),
        ("gemini", "g", "o", GeminiEmbedder),
        ("gemini", None, "o", None),
        ("openai", "g", "o", OpenAIEmbedder),
        ("openai", "g", None, None),
    ],
)
def test_get_embedder_explicit_provider(provider, gemini_key, openai_key, expected, monkeypatch):
    monkeypatch.setattr(settings, "router_embedding_provider", provider)
    monkeypatch.setattr(settings, "gemini_api_key", gemini_key)
    monkeypatch.setattr(settings, "openai_api_key", openai_key)

    embedder = get_embedder()
    if expected is None:
        assert embedder is None
    else:
        assert isinstance(embedder, expected)


@pytest.mark.parametrize(
    ("gemini_key", "openai_key", "expected"),
    [
        ("g", "o", GeminiEmbedder),  # auto prefers the owner's working provider
        (None, "o", OpenAIEmbedder),
        (None, None, None),
    ],
)
def test_get_embedder_auto_prefers_gemini(gemini_key, openai_key, expected, monkeypatch):
    monkeypatch.setattr(settings, "router_embedding_provider", "auto")
    monkeypatch.setattr(settings, "gemini_api_key", gemini_key)
    monkeypatch.setattr(settings, "openai_api_key", openai_key)

    embedder = get_embedder()
    if expected is None:
        assert embedder is None
    else:
        assert isinstance(embedder, expected)


def test_embedder_model_defaults():
    assert OpenAIEmbedder()._model == "text-embedding-3-small"
    assert OpenAIEmbedder("my-model")._model == "my-model"
    assert GeminiEmbedder()._model == "models/text-embedding-004"
