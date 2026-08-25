"""Session inspection endpoints for working memory."""

from fastapi import APIRouter, HTTPException

from app.api.schemas import SessionDetail, SessionTurn
from app.config import settings
from app.memory import session_store
from app.rediskit import holder

router = APIRouter()


@router.get("/session/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    if not settings.memory_enabled:
        raise HTTPException(status_code=503, detail="Memory is disabled.")
    if holder.client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable; memory offline.")

    turns = await session_store.recent(session_id, limit=settings.memory_max_turns)
    return SessionDetail(
        session_id=session_id,
        turns=[SessionTurn(**turn) for turn in turns],
        enabled=True,
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    from app.memory import session_store

    cleared = await session_store.clear(session_id)
    return {"session_id": session_id, "cleared": cleared}
