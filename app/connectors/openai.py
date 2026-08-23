import asyncio
import time

from app.config import settings
from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
    classify_provider_exception,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIConnector(BaseConnector):
    connector_id = "openai"
    display_name = "OpenAI GPT-4o"
    capabilities = ["text", "vision", "code", "structured_output", "instruction_following"]

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.default_model = "gpt-4o"
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
                error="OPENAI_API_KEY not configured",
                sub_query=sub_query,
            )

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a specialized AI assistant. "
                                "Answer only the specific task given to you. "
                                "The original user query is provided for context only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Original user query (context): {prompt}\n\n"
                                f"Your specific task: {sub_query}"
                            ),
                        },
                    ],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                ),
                timeout=config.timeout_s,
            )

            latency_ms = int((time.monotonic() - start) * 1000)
            content = response.choices[0].message.content or ""

            raw_usage = response.usage
            usage = TokenUsage(
                prompt_tokens=raw_usage.prompt_tokens if raw_usage else 0,
                completion_tokens=raw_usage.completion_tokens if raw_usage else 0,
                total_tokens=raw_usage.total_tokens if raw_usage else 0,
            )

            return ConnectorResponse(
                model_id=model,
                content=content,
                latency_ms=latency_ms,
                token_usage=usage,
                status=ConnectorStatus.SUCCESS,
                sub_query=sub_query,
            )

        except TimeoutError:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning({"message": "OpenAI timeout", "latency_ms": latency_ms})
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
            status, retry_after_s = classify_provider_exception(e)
            logger.error({"message": "OpenAI error", "error": str(e), "status": status.value})
            return ConnectorResponse(
                model_id=model,
                content="",
                latency_ms=latency_ms,
                token_usage=TokenUsage(),
                status=status,
                error=str(e),
                sub_query=sub_query,
                retry_after_s=retry_after_s,
            )

    async def health_check(self) -> bool:
        return bool(self.api_key)

    async def stream_query(
        self,
        prompt: str,
        sub_query: str,
        config: ConnectorConfig,
    ):
        """Native token streaming for OpenAI chat completions."""
        if not self.api_key:
            return
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        model = config.model_override or self.default_model
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a specialized AI assistant. "
                        "Answer only the specific task given to you. "
                        "The original user query is provided for context only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original user query (context): {prompt}\n\n"
                        f"Your specific task: {sub_query}"
                    ),
                },
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stream=True,
        )
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await stream.close()

