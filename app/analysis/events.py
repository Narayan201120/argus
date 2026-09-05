"""P4-3 investigation event bus (DEC-053, backend only).

Per-investigation in-memory pub/sub for live tails (SSE). Single process
only: restarts drop live tails, and snapshot-on-join covers recovery
(new subscribers reload board/report snapshots via the stores).
"""

import asyncio
from typing import Any

from app.metrics import BUS_DROPS
from app.utils.logger import get_logger

logger = get_logger(__name__)

BOARD_SNAPSHOT = "board_snapshot"
ROUND_STARTED = "round_started"
EVIDENCE_ADDED = "evidence_added"
CLAIMS_UPDATED = "claims_updated"
SYNTHESIS_START = "synthesis_start"
SYNTHESIS_TOKEN = "synthesis_token"
SYNTHESIS_END = "synthesis_end"
TERMINAL = "terminal"

_subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}


def subscribe(investigation_id: str) -> asyncio.Queue[dict[str, Any]]:
    """Attach a bounded live tail for one investigation."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
    _subscribers.setdefault(investigation_id, []).append(queue)
    return queue


def unsubscribe(
    investigation_id: str, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    """Detach a live tail; drops empty per-investigation keys."""
    try:
        current = _subscribers.get(investigation_id)
        if not current:
            return
        try:
            current.remove(queue)
        except ValueError:
            return
        if not current:
            _subscribers.pop(investigation_id, None)
    except Exception as exc:  # noqa: BLE001 - unsubscribe never raises
        logger.warning(
            {
                "message": "Event bus unsubscribe failed (ignored)",
                "investigation_id": investigation_id,
                "error": str(exc),
            }
        )


async def publish(investigation_id: str, event: str, data: dict[str, Any]) -> None:
    """Fan out one event to every live tail. Never raises.

    Slow subscribers drop the item (bounded queue) and count one
    BUS_DROPS increment each.
    """
    try:
        targets = list(_subscribers.get(investigation_id) or [])
        for queue in targets:
            try:
                queue.put_nowait({"event": event, "data": data})
            except asyncio.QueueFull:
                try:
                    BUS_DROPS.labels(event=event).inc()
                except Exception as exc:  # noqa: BLE001 - metrics must not break the bus
                    logger.warning(
                        {
                            "message": "Event bus drop metric failed (ignored)",
                            "investigation_id": investigation_id,
                            "error": str(exc),
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - one bad subscriber must not break fan-out
                logger.warning(
                    {
                        "message": "Event bus publish failed for subscriber (ignored)",
                        "investigation_id": investigation_id,
                        "error": str(exc),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - publish never raises
        logger.warning(
            {
                "message": "Event bus publish failed (ignored)",
                "investigation_id": investigation_id,
                "error": str(exc),
            }
        )
