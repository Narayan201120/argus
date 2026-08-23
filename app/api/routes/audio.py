"""Audio input: Sarvam STT transcription and voice-driven queries.

POST /v1/transcribe   - audio file -> text (edit-friendly transcript)
POST /v1/query/audio  - audio file -> transcript -> normal query pipeline

Both endpoints enforce the credit-safety guards from DEC-040: format
allowlist, upload size cap, no provider retries, and clean 503s when
Sarvam is not configured or disabled.
"""

import json
import time
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.routes.query import run_query
from app.api.schemas import AudioQueryResponse, ConnectorConfigRequest, QueryRequest, TranscriptionResponse
from app.config import settings
from app.metrics import TRANSCRIPTION_LATENCY, TRANSCRIPTIONS
from app.tracing import span
from app.transcription import TranscriptionError, get_transcriber, validate_upload
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


async def _transcribe_upload(file: UploadFile) -> TranscriptionResponse:
    """Shared guard chain + provider call for both audio endpoints."""
    transcriber = get_transcriber()
    if not settings.transcription_enabled or transcriber is None:
        TRANSCRIPTIONS.labels(status="unavailable").inc()
        raise HTTPException(
            status_code=503,
            detail="Audio transcription is not configured (SARVAM_API_KEY missing or disabled).",
        )

    try:
        validate_upload(file.filename)
    except TranscriptionError as exc:
        TRANSCRIPTIONS.labels(status="rejected").inc()
        raise HTTPException(status_code=exc.status_code or 415, detail=str(exc)) from exc

    content = await file.read(settings.audio_max_upload_bytes + 1)
    if len(content) > settings.audio_max_upload_bytes:
        TRANSCRIPTIONS.labels(status="rejected").inc()
        limit_mb = settings.audio_max_upload_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {limit_mb:.0f} MB upload cap.",
        )

    start = time.monotonic()
    try:
        with span(
            "transcription.sarvam",
            {"argus.audio_bytes": len(content), "argus.audio_model": settings.sarvam_stt_model},
        ) as current:
            result = await transcriber.transcribe(content, file.filename or "audio", file.content_type)
            if current is not None:
                current.set_attribute("argus.text_length", len(result.text))
    except TranscriptionError as exc:
        latency = time.monotonic() - start
        TRANSCRIPTION_LATENCY.observe(latency)
        if exc.status_code == 429:
            TRANSCRIPTIONS.labels(status="rate_limited").inc()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        TRANSCRIPTIONS.labels(status="error").inc()
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latency = time.monotonic() - start
    TRANSCRIPTION_LATENCY.observe(latency)
    TRANSCRIPTIONS.labels(status="success").inc()
    logger.info({
        "message": "Audio transcribed",
        "filename": file.filename,
        "bytes": len(content),
        "latency_ms": result.latency_ms,
        "text_length": len(result.text),
    })
    return TranscriptionResponse(
        text=result.text,
        language_code=result.language_code,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: Annotated[UploadFile, File(...)]) -> TranscriptionResponse:
    return await _transcribe_upload(file)


@router.post("/query/audio", response_model=AudioQueryResponse)
async def query_audio(
    file: Annotated[UploadFile, File(...)],
    options: Annotated[str | None, Form()] = None,
) -> AudioQueryResponse:
    """Transcribe the upload, then answer it through the standard pipeline.

    ``options`` optionally carries the same model_config object accepted
    by /v1/query (connectors, profile, router_strategy, timeout_s, ...).
    """
    transcription = await _transcribe_upload(file)

    model_config_request = ConnectorConfigRequest()
    if options:
        try:
            parsed = json.loads(options)
            if not isinstance(parsed, dict):
                raise ValueError("options must be a JSON object")
            model_config_request = ConnectorConfigRequest.model_validate(parsed)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid options JSON: {exc}") from exc

    request = QueryRequest(query=transcription.text)
    request.model_config_ = model_config_request
    response = await run_query(request)

    return AudioQueryResponse(
        **response.model_dump(),
        transcript_text=transcription.text,
        transcript_language_code=transcription.language_code,
        transcript_model=transcription.model,
        transcription_latency_ms=transcription.latency_ms,
    )
