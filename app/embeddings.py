"""Pluggable text-embedding backends for the semantic router.

Only two tiny implementations exist (OpenAI, Gemini) because those SDKs
are already required by the connectors - this module adds no new
dependencies. Embeddings are used solely to match a query against
routing-profile descriptions; plain cosine similarity over a handful of
vectors needs no vector database.
"""

from abc import ABC, abstractmethod

from app.config import settings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-python cosine between equal-length vectors (0.0 if degenerate)."""
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class BaseEmbedder(ABC):
    """Minimal async embedding interface."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str | None = None):
        self._model = model or "text-embedding-3-small"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class GeminiEmbedder(BaseEmbedder):
    def __init__(self, model: str | None = None):
        self._model = model or "text-embedding-004"
        self._api_key = settings.gemini_api_key

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        from google import genai

        client = genai.Client(api_key=self._api_key)
        result = await asyncio.to_thread(
            client.models.embed_content, model=self._model, contents=texts
        )
        embedding_objects = getattr(result, "embeddings", None) or []
        return [list(item.values) for item in embedding_objects]


def get_embedder() -> BaseEmbedder | None:
    """Pick an embedder per ROUTER_EMBEDDING_PROVIDER.

    'auto' prefers Gemini whenever its key exists (the owner's working
    provider); OpenAI is chosen only if it is the sole key present.
    Returns None when embeddings are disabled/unavailable, which makes
    the semantic router degrade to the keyword classifier.
    """
    provider = settings.router_embedding_provider.lower()
    if provider == "none":
        return None
    if provider == "openai":
        return OpenAIEmbedder(settings.router_embedding_model) if settings.openai_api_key else None
    if provider == "gemini":
        return GeminiEmbedder(settings.router_embedding_model) if settings.gemini_api_key else None

    # auto
    if settings.gemini_api_key:
        return GeminiEmbedder(settings.router_embedding_model)
    if settings.openai_api_key:
        return OpenAIEmbedder(settings.router_embedding_model)
    return None
