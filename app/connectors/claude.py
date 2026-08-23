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


class ClaudeConnector(BaseConnector):
    connector_id = "claude"
    display_name = "Anthropic Claude"
    capabilities = ["text", "reasoning", "synthesis", "long_form", "document"]

    def __init__(self):
        self.api_key = settings.anthropic_api_key
        self.default_model = "claude-sonnet-4-5"
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
                error="ANTHROPIC_API_KEY not configured",
                sub_query=sub_query,
            )

        try:
            import anthropic
            from anthropic.types import TextBlock

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            response = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=config.max_tokens,
                    system=(
                        "You are a specialized AI assistant. "
                        "Answer only the specific task given to you. "
                        "The original user query is provided for context only."
                    ),
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Original user query (context): {prompt}\n\n"
                                f"Your specific task: {sub_query}"
                            ),
                        }
                    ],
                ),
                timeout=config.timeout_s,
            )

            latency_ms = int((time.monotonic() - start) * 1000)
            content = "".join(
                block.text for block in response.content if isinstance(block, TextBlock)
            )

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
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
            logger.warning({"message": "Claude timeout", "latency_ms": latency_ms})
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
            logger.error({"message": "Claude error", "error": str(e), "status": status.value})
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
