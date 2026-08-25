"""Sarvam Bulbul text-to-speech.

API facts (docs.sarvam.ai, verified Aug 2026):
- POST https://api.sarvam.ai/text-to-speech, application/json
- Auth header: ``api-subscription-key`` (NOT Authorization: Bearer)
- Body: ``text``, ``model`` (bulbul:v3 recommended), ``speaker``
  (lowercase, must match model version), ``target_language_code``
- Response: ``{"request_id": ..., "audios": ["<base64 WAV>", ...]}``
- bulbul:v3 char limit: 2500 (we default lower for credit care)
"""

import base64
import time

import httpx

from app.config import settings
from app.transcription.base import TranscriptionError

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"


def get_tts():
    """Factory keyed off SARVAM_API_KEY + SPEECH_ENABLED."""
    if settings.sarvam_api_key and settings.speech_enabled:
        return SarvamTTS()
    return None


class SarvamTTS:
    async def synthesize(self, text: str) -> bytes:
        """Return WAV bytes for the given text (single-audio response)."""
        start = time.monotonic()
        headers = {"api-subscription-key": settings.sarvam_api_key or ""}
        payload = {
            "text": text,
            "target_language_code": settings.sarvam_tts_language,
            "speaker": settings.sarvam_tts_speaker,
            "model": settings.sarvam_tts_model,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.speech_timeout_s) as client:
                response = await client.post(SARVAM_TTS_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"Sarvam TTS request failed: {exc}") from exc

        if response.status_code != 200:
            raise TranscriptionError(
                f"Sarvam TTS HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        audios = (response.json() or {}).get("audios") or []
        if not audios:
            raise TranscriptionError("Sarvam TTS returned no audio.")

        _ = start  # latency tracked by the route-level histogram
        return base64.b64decode(audios[0])
