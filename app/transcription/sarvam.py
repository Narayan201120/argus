"""Sarvam AI speech-to-text transcriber.

API facts (docs.sarvam.ai, verified Aug 2026):
- POST https://api.sarvam.ai/speech-to-text, multipart/form-data
- Auth header: ``api-subscription-key`` (NOT Authorization: Bearer)
- model ``saaras:v3`` recommended; mode transcribe|translate|verbatim|translit|codemix
- REST hard cap: 30 seconds of audio per file (batch API deferred)
"""

import time

import httpx

from app.config import settings
from app.transcription.base import BaseTranscriber, TranscriptionError, TranscriptionResult

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamTranscriber(BaseTranscriber):
    async def transcribe(
        self, content: bytes, filename: str, content_type: str | None
    ) -> TranscriptionResult:
        start = time.monotonic()
        headers = {"api-subscription-key": settings.sarvam_api_key or ""}
        data = {
            "model": settings.sarvam_stt_model,
            "mode": settings.sarvam_stt_mode,
            "language_code": settings.sarvam_stt_language,
        }
        files = {"file": (filename, content, content_type or "application/octet-stream")}

        try:
            async with httpx.AsyncClient(timeout=settings.transcription_timeout_s) as client:
                response = await client.post(SARVAM_STT_URL, headers=headers, data=data, files=files)
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Sarvam request failed: {exc}") from exc

        if response.status_code != 200:
            raise TranscriptionError(
                f"Sarvam HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        payload = response.json()
        text = str(payload.get("transcript") or payload.get("text") or "").strip()
        if not text:
            raise TranscriptionError("Sarvam returned an empty transcript.")

        return TranscriptionResult(
            text=text,
            language_code=payload.get("language_code"),
            model=str(payload.get("model") or settings.sarvam_stt_model),
            latency_ms=int((time.monotonic() - start) * 1000),
        )
