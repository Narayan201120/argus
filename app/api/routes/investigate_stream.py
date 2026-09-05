"""SSE board-stream endpoint for investigations (P4-3, backend only, mock-only).

GET /investigate/{investigation_id}/stream replays the full board snapshot
first (BOARD_SNAPSHOT, the same envelope as GET /investigate/{id} including
syntheses) then tails the investigation event bus until the TERMINAL event.
Unknown ids 404 before streaming starts. Client disconnect ends the
generator; unsubscribe always runs in the finally block.

Router registration in app/main.py belongs to the orchestrator, not this file.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from app.analysis import events as events_module
from app.api.routes.investigations import read_investigation
from app.investigations import TERMINAL
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _sse(event: str, data: dict[str, Any]) -> str:
    # Same wire format as app/api/routes/stream.py; kept local so this route
    # does not import the query-stream pipeline.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/investigate/{investigation_id}/stream")
async def stream_investigation(investigation_id: str) -> StreamingResponse:
    board = await read_investigation(investigation_id)  # 404 when unknown.
    snapshot = board.model_dump(mode="json")

    async def event_stream() -> AsyncIterator[str]:
        yield _sse(events_module.BOARD_SNAPSHOT, snapshot)
        if board.status in TERMINAL:
            # Late subscriber: the TERMINAL broadcast already went out before
            # this tail started, so replay it from the snapshot and finish
            # instead of hanging on a bus that will never speak again.
            reason = (
                board.status_reason.value
                if board.status_reason is not None
                else board.status.value
            )
            yield _sse(
                events_module.TERMINAL,
                {"status": board.status.value, "reason": reason},
            )
            return
        try:
            queue = events_module.subscribe(investigation_id)
        except Exception as exc:  # noqa: BLE001 - snapshot already yielded; just end
            logger.warning(
                {
                    "message": "Investigation stream: subscribe failed",
                    "investigation_id": investigation_id,
                    "error": str(exc),
                }
            )
            return
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if not isinstance(item, dict) or "event" not in item:
                    continue
                event = str(item["event"])
                data = item.get("data", {})
                if not isinstance(data, dict):
                    data = {"value": data}
                yield _sse(event, data)
                if event == events_module.TERMINAL:
                    break
        finally:
            try:
                events_module.unsubscribe(investigation_id, queue)
            except Exception as exc:  # noqa: BLE001 - unsubscribe is best effort
                logger.warning(
                    {
                        "message": "Investigation stream: unsubscribe failed",
                        "investigation_id": investigation_id,
                        "error": str(exc),
                    }
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
