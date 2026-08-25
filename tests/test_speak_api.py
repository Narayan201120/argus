"""Stage P3-2 - /v1/speak endpoint (mock-only, no Sarvam calls)."""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.transcription.base import TranscriptionError

client = TestClient(app)

WAV_BYTES = b"RIFF....fake"


class StubTTS:
    def __init__(self, audio: bytes = WAV_BYTES, error=None):
        self._audio = audio
        self._error = error

    async def synthesize(self, text: str) -> bytes:
        if self._error is not None:
            raise self._error
        return self._audio


def _use_tts(monkeypatch, tts):
    monkeypatch.setattr(settings, "speech_enabled", True)
    monkeypatch.setattr("app.api.routes.audio.get_tts", lambda: tts)


def test_speak_success_returns_wav(monkeypatch):
    _use_tts(monkeypatch, StubTTS())
    response = client.post("/v1/speak", json={"text": "Hello from ARGUS."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


def test_speak_disabled_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "speech_enabled", False)
    monkeypatch.setattr("app.api.routes.audio.get_tts", lambda: None)
    response = client.post("/v1/speak", json={"text": "Hello"})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_speak_rejects_empty_text(monkeypatch):
    _use_tts(monkeypatch, StubTTS())
    response = client.post("/v1/speak", json={"text": "   "})
    assert response.status_code == 422


def test_speak_enforces_character_cap(monkeypatch):
    _use_tts(monkeypatch, StubTTS())
    monkeypatch.setattr(settings, "speech_max_chars", 10)
    response = client.post("/v1/speak", json={"text": "x" * 50})
    assert response.status_code == 422
    assert "character speech cap" in response.json()["detail"]


def test_speak_maps_provider_429(monkeypatch):
    _use_tts(
        monkeypatch,
        StubTTS(error=TranscriptionError("TTS quota exhausted", status_code=429)),
    )
    response = client.post("/v1/speak", json={"text": "Hello"})
    assert response.status_code == 429


def test_speak_maps_provider_error_to_502(monkeypatch):
    _use_tts(monkeypatch, StubTTS(error=TranscriptionError("Sarvam TTS HTTP 500", 500)))
    response = client.post("/v1/speak", json={"text": "Hello"})
    assert response.status_code == 502


def test_speech_metrics_recorded(monkeypatch):
    from prometheus_client import REGISTRY

    _use_tts(monkeypatch, StubTTS())
    ok = client.post("/v1/speak", json={"text": "Hello"})
    assert ok.status_code == 200

    value = REGISTRY.get_sample_value("argus_speech_total", {"status": "success"})
    assert value is not None and value >= 1.0
