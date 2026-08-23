"""Transcription contracts shared by providers and routes."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import settings

# Sarvam REST accepts these containers for /speech-to-text (30s audio cap).
ALLOWED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".oga", ".webm", ".flac"})
AUDIO_MIME_PREFIX = "audio/"


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language_code: str | None
    model: str
    latency_ms: int


class TranscriptionError(Exception):
    """Provider-side transcription failure.

    status_code carries the upstream HTTP status when known so routes can
    pass 429 through instead of masking quota issues as generic errors.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def validate_upload(filename: str | None) -> str:
    """Return the normalized extension or raise TranscriptionError(415)."""
    if not filename:
        raise TranscriptionError("Uploaded file has no name; cannot verify audio format.", 415)
    dot = filename.rfind(".")
    extension = filename[dot:].lower() if dot != -1 else ""
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise TranscriptionError(
            f"Unsupported audio format '{extension or filename}'. Allowed: {allowed}.",
            415,
        )
    return extension


class BaseTranscriber(ABC):
    """Minimal async speech-to-text interface."""

    @abstractmethod
    async def transcribe(
        self, content: bytes, filename: str, content_type: str | None
    ) -> TranscriptionResult:
        """Transcribe raw audio bytes and return the result."""


def get_transcriber() -> BaseTranscriber | None:
    """Factory keyed off SARVAM_API_KEY. None means STT is not configured."""
    from app.transcription.sarvam import SarvamTranscriber

    if settings.sarvam_api_key and settings.transcription_enabled:
        return SarvamTranscriber()
    return None


def monotonic_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
