import asyncio
import time
from typing import Optional

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiConnector(BaseConnector):
    connector_id = "gemini"
    display_name = "Google Gemini"
    capabilities = ["text", "vision", "research", "pdf", "long_context"]

    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.default_model = "gemini-1.5-pro"
        self.is_available = bool(self.api_key)

    async def query(
        self,
        prompt: str,
        sub_query: str,
        config: ConnectorConfig,
    ) -> ConnectorResponse:
        start = time.monotonic()
        model = config.model_override or self.default_model

        if not self.api_key:
            return ConnectorResponse(
                model_id=model,
                content="",
                latency_ms=0,
                token_usage=TokenUsage(),
                status=ConnectorStatus.ERROR,
                error="GEMINI_API_KEY not configured",
                sub_query=sub_query,
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            client = genai.GenerativeModel(model)

            full_prompt = (
                f"Original user query (for context only): {prompt}\n\n"
                f"Your specific task: {sub_query}"
            )

            response = await asyncio.wait_for(
                asyncio.to_thread(client.generate_content, full_prompt),
                timeout=config.timeout_s,
            )

            latency_ms = int((time.monotonic() - start) * 1000)
            content = response.text or ""

            usage = TokenUsage()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage.prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                usage.completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                usage.total_tokens = getattr(response.usage_metadata, "total_token_count", 0)

            return ConnectorResponse(
                model_id=model,
                content=content,
                latency_ms=latency_ms,
                token_usage=usage,
                status=ConnectorStatus.SUCCESS,
                sub_query=sub_query,
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning({"message": "Gemini timeout", "latency_ms": latency_ms})
            return ConnectorResponse(
                model_id=model,
                content="",
                latency_ms=latency_ms,
                token_usage=TokenUsage(),
                status=ConnectorStatus.TIMEOUT,
                error=f"Timed out after {config.timeout_s}s",
                sub_query=sub_query,
            )

        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error({"message": "Gemini error", "error": str(e)})
            return ConnectorResponse(
                model_id=model,
                content="",
                latency_ms=latency_ms,
                token_usage=TokenUsage(),
                status=ConnectorStatus.ERROR,
                error=str(e),
                sub_query=sub_query,
            )

    async def health_check(self) -> bool:
        return bool(self.api_key)
