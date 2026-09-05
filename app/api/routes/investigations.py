"""Async investigation endpoints (P4-0 routes only, no business logic)."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.analysis.loop import run_investigation_loop
from app.analysis.synthesis import synthesis_store
from app.api.schemas import (
    BoardCounts,
    CancelInvestigationResponse,
    InvestigateCreated,
    InvestigateRequest,
    InvestigationBoardResponse,
)
from app.investigations import manager

router = APIRouter()


@router.post("/investigate", response_model=InvestigateCreated, status_code=202)
async def start_investigation(
    request: InvestigateRequest, background_tasks: BackgroundTasks
) -> InvestigateCreated:
    try:
        investigation = await manager.create(request.query.strip(), request.user_id.strip() or "local")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    background_tasks.add_task(run_investigation_loop, investigation.id)
    return InvestigateCreated(
        investigation_id=investigation.id,
        user_id=investigation.user_id,
        status=investigation.status,
        status_reason=investigation.status_reason,
    )


@router.get("/investigate/{investigation_id}", response_model=InvestigationBoardResponse)
async def read_investigation(investigation_id: str) -> InvestigationBoardResponse:
    investigation = await manager.get(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    evidence = list(investigation.board.evidence)
    claims = list(investigation.board.claims)
    syntheses = await synthesis_store.load(investigation_id)
    return InvestigationBoardResponse(
        investigation_id=investigation.id,
        user_id=investigation.user_id,
        query=investigation.query,
        status=investigation.status,
        status_reason=investigation.status_reason,
        created_at=investigation.created_at,
        updated_at=investigation.updated_at,
        schema_version=investigation.schema_version,
        evidence=evidence,
        claims=claims,
        counts=BoardCounts(evidence=len(evidence), claims=len(claims)),
        truncated=False,
        syntheses=syntheses,
    )


@router.post("/investigate/{investigation_id}/cancel", response_model=CancelInvestigationResponse)
async def cancel_investigation(investigation_id: str) -> CancelInvestigationResponse:
    investigation = await manager.cancel(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return CancelInvestigationResponse(
        investigation_id=investigation.id,
        user_id=investigation.user_id,
        status=investigation.status,
        status_reason=investigation.status_reason,
    )
