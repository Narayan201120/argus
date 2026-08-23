"""Stage P2-2 - audio transcription endpoints (mock-only, no Sarvam calls)."""

from fastapi.testclient import TestClient

from app.config import settings
from app.connectors.base import BaseConnector, ConnectorResponse, ConnectorStatus, TokenUsage
from app.connectors.registry import registry
from app.main import app
from app.transcription.base import (
    BaseTranscriber,
    TranscriptionError,
    TranscriptionResult,
)

client = TestClient(app)

WAV_BYTES = b"RIFF....fake wav payload"


class StubTranscriber(BaseTranscriber):
    def __init__(self, result=None, error=None):
        self._result = result or TranscriptionResult(
            text="what is ARGUS",
            language_code="en-IN",
            model="saaras:v3",
            latency_ms=42,
        )
        self._error = error

    async def transcribe(self, content, filename, content_type):
        if self._error is not None:
            raise self._error
        return self._result


class StubConnector(BaseConnector):
    capabilities = ["text"]
    is_available = True

    def __init__(self, connector_id: str):
        self.connector_id = connector_id
        self.display_name = f"{connector_id.title()} Stub"

    async def query(self, prompt, sub_query, config):
        return ConnectorResponse(
            model_id=self.connector_id,
            content="Direct response",
            latency_ms=1,
            token_usage=TokenUsage(1, 1, 2),
            status=ConnectorStatus.SUCCESS,
            sub_query=sub_query,
        )

    async def health_check(self):
        return True


def _use_transcriber(monkeypatch, transcriber):
    monkeypatch.setattr(settings, "transcription_enabled", True)
    monkeypatch.setattr(
        "app.api.routes.audio.get_transcriber", lambda: transcriber
    )


def test_transcribe_success(monkeypatch):
    _use_transcriber(monkeypatch, StubTranscriber())
    response = client.post(
        "/v1/transcribe", files={"file": ("question.wav", WAV_BYTES, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "what is ARGUS"
    assert data["model"] == "saaras:v3"
    assert data["language_code"] == "en-IN"


def test_transcribe_disabled_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "transcription_enabled", False)
    monkeypatch.setattr("app.api.routes.audio.get_transcriber", lambda: None)
    response = client.post(
        "/v1/transcribe", files={"file": ("question.wav", WAV_BYTES, "audio/wav")}
    )
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_transcribe_rejects_bad_extension(monkeypatch):
    _use_transcriber(monkeypatch, StubTranscriber())
    response = client.post(
        "/v1/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415


def test_transcribe_rejects_oversized_upload(monkeypatch):
    _use_transcriber(monkeypatch, StubTranscriber())
    monkeypatch.setattr(settings, "audio_max_upload_bytes", 8)
    response = client.post(
        "/v1/transcribe", files={"file": ("big.wav", b"x" * 64, "audio/wav")}
    )
    assert response.status_code == 413


def test_transcribe_maps_provider_429(monkeypatch):
    _use_transcriber(
        monkeypatch,
        StubTranscriber(error=TranscriptionError("quota exhausted", status_code=429)),
    )
    response = client.post(
        "/v1/transcribe", files={"file": ("q.wav", WAV_BYTES, "audio/wav")}
    )
    assert response.status_code == 429


def test_transcribe_maps_provider_error_to_502(monkeypatch):
    _use_transcriber(
        monkeypatch, StubTranscriber(error=TranscriptionError("Sarvam HTTP 500: boom", 500))
    )
    response = client.post(
        "/v1/transcribe", files={"file": ("q.wav", WAV_BYTES, "audio/wav")}
    )
    assert response.status_code == 502


def test_query_audio_answers_through_pipeline(monkeypatch):
    _use_transcriber(monkeypatch, StubTranscriber())
    monkeypatch.setattr(registry, "_connectors", {"mistral": StubConnector("mistral")})

    response = client.post(
        "/v1/query/audio",
        files={"file": ("question.wav", WAV_BYTES, "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    # transcript fed into the standard pipeline
    assert data["query"] == "what is ARGUS"
    assert data["result"] == "Direct response"
    assert data["role_assignments"]["direct"] == "mistral"
    assert data["transcript_text"] == "what is ARGUS"
    assert data["transcript_model"] == "saaras:v3"


def test_query_audio_forwards_options_json(monkeypatch):
    captured = {}
    _use_transcriber(monkeypatch, StubTranscriber())

    class SpyConnector(StubConnector):
        async def query(self, prompt, sub_query, config):
            captured["timeout_s"] = config.timeout_s
            return await super().query(prompt, sub_query, config)

    monkeypatch.setattr(registry, "_connectors", {"mistral": SpyConnector("mistral")})

    response = client.post(
        "/v1/query/audio",
        files={"file": ("question.wav", WAV_BYTES, "audio/wav")},
        data={"options": '{"connectors": ["mistral"], "timeout_s": 90}'},
    )

    assert response.status_code == 200
    assert captured["timeout_s"] == 90


def test_query_audio_rejects_bad_options_json(monkeypatch):
    _use_transcriber(monkeypatch, StubTranscriber())
    response = client.post(
        "/v1/query/audio",
        files={"file": ("question.wav", WAV_BYTES, "audio/wav")},
        data={"options": "not-json"},
    )
    assert response.status_code == 422


def test_transcription_metrics_visible_in_scrape(monkeypatch):
    from prometheus_client import REGISTRY

    before = REGISTRY.get_sample_value(
        "argus_transcriptions_total", {"status": "success"}
    ) or 0.0

    _use_transcriber(monkeypatch, StubTranscriber())
    ok = client.post("/v1/transcribe", files={"file": ("q.wav", WAV_BYTES, "audio/wav")})
    rejected = client.post("/v1/transcribe", files={"file": ("x.txt", b"nope", "text/plain")})
    assert ok.status_code == 200
    assert rejected.status_code == 415

    after = REGISTRY.get_sample_value(
        "argus_transcriptions_total", {"status": "success"}
    )
    rejected_count = REGISTRY.get_sample_value(
        "argus_transcriptions_total", {"status": "rejected"}
    )
    assert after is not None and after > before
    assert rejected_count is not None and rejected_count >= 1.0
