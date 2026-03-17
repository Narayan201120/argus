import asyncio
import time
from app.connectors.base import (
    BaseConnector, ConnectorConfig, ConnectorResponse, ConnectorStatus, TokenUsage,
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class MistralConnector(BaseConnector):
    connector_id = "mistral"
    display_name = "Mistral AI"
    capabilities = ["text", "code", "reasoning", "instruction_following"]

    def __init__(self):
        self.api_key = settings.mistral_api_key
        self.default_model = "mistral-small-latest"
        self.is_available = bool(self.api_key)

    async def query(self, prompt, sub_query, config):
        start = time.monotonic()
        model = config.model_override or self.default_model

        if not self.api_key:
            return ConnectorResponse(
                model_id=model,
                content="",
                latency_ms=0,
                token_usage=TokenUsage(),
                status=ConnectorStatus.ERROR,
                error="MISTRAL_API_KEY not configured",
                sub_query=sub_query,
            )

        try:
            from mistralai import Mistral

            client = Mistral(api_key=self.api_key)

            response = await asyncio.wait_for(
                    client.chat.complete_async(
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

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

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
            logger.warning({"message": "Mistral timeout", "latency_ms": latency_ms})
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
            logger.error({"message": "Mistral error", "error": str(e)})
            return ConnectorResponse(
                model_id=model,
                content="",
                latency_ms=latency_ms,
                token_usage=TokenUsage(),
                status=ConnectorStatus.ERROR,
                error=str(e),
                sub_query=sub_query,
            )
    async def health_check(self):
        return bool(self.api_key)