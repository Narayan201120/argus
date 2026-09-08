"""Quality feedback endpoint for A/B routing experiments."""

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import FeedbackRequest, FeedbackResponse
from app.auth import resolve_subject
from app.feedback import get_rating, save_rating
from app.metrics import FEEDBACK_TOTAL

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(
    request: FeedbackRequest, http_request: Request
) -> FeedbackResponse:
    stored = await save_rating(
        request.request_id, request.rating, owner=resolve_subject(http_request)
    )
    if not stored:
        raise HTTPException(
            status_code=503,
            detail="Feedback storage unavailable (Redis down or memory disabled).",
        )
    FEEDBACK_TOTAL.labels(rating=str(request.rating)).inc()
    return FeedbackResponse(request_id=request.request_id, rating=request.rating, stored=True)


@router.get("/feedback/{request_id}", response_model=FeedbackResponse)
async def read_feedback(request_id: str, http_request: Request) -> FeedbackResponse:
    from fastapi import HTTPException as _HTTPException

    rating = await get_rating(request_id, owner=resolve_subject(http_request))
    if rating is None:
        raise _HTTPException(status_code=404, detail="No feedback stored for this request.")
    return FeedbackResponse(request_id=request_id, rating=rating, stored=True)
