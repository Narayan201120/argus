import asyncio
import time

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorResponse,
    ConnectorStatus,
    TokenUsage,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def _query_with_retry(
    connector: BaseConnector,
    prompt: str,
    sub_query: str,
    config: ConnectorConfig,
) -> ConnectorResponse:
    """Run a connector query with one automatic retry on timeout."""
    response = await connector.query(prompt, sub_query, config)

    if response.status == ConnectorStatus.TIMEOUT and config.max_retries > 0:
        logger.info({
            "message": "Retrying connector after timeout",
            "connector_id": connector.connector_id,
        })
        retry_config = ConnectorConfig(
            timeout_s=config.timeout_s,
            max_retries=0,  # No further retries
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        response = await connector.query(prompt, sub_query, retry_config)

    return response


async def dispatch(
    prompt: str,
    sub_queries: dict[str, str],          # connector_id -> sub_query
    connectors: dict[str, BaseConnector],  # connector_id -> connector instance
    config: ConnectorConfig | None = None,
) -> dict[str, ConnectorResponse]:
    """
    Fan out sub-queries to all connectors concurrently via asyncio.gather.

    Uses return_exceptions=True so a single connector failure never
    blocks the rest. Failed connectors are wrapped in error ConnectorResponses.

    Returns:
        dict[connector_id -> ConnectorResponse] for all dispatched connectors.
    """
    if config is None:
        config = ConnectorConfig()

    start = time.monotonic()

    # Build coroutine map — only dispatch connectors that exist and are available
    tasks: dict[str, asyncio.coroutine] = {
        cid: _query_with_retry(
            connector=connectors[cid],
            prompt=prompt,
            sub_query=sub_queries[cid],
            config=config,
        )
        for cid in sub_queries
        if cid in connectors and connectors[cid].is_available
    }

    if not tasks:
        logger.warning({"message": "No connectors available to dispatch to"})
        return {}

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    response_bundle: dict[str, ConnectorResponse] = {}
    for connector_id, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.error({
                "message": "Unhandled connector exception",
                "connector_id": connector_id,
                "error": str(result),
            })
            response_bundle[connector_id] = ConnectorResponse(
                model_id=connector_id,
                content="",
                latency_ms=int((time.monotonic() - start) * 1000),
                token_usage=TokenUsage(),
                status=ConnectorStatus.ERROR,
                error=str(result),
            )
        else:
            response_bundle[connector_id] = result

    total_ms = int((time.monotonic() - start) * 1000)
    succeeded = sum(
        1 for r in response_bundle.values() if r.status == ConnectorStatus.SUCCESS
    )
    logger.info({
        "message": "Dispatch complete",
        "succeeded": succeeded,
        "total": len(tasks),
        "total_ms": total_ms,
    })

    return response_bundle
