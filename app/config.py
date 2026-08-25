
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AI Provider API Keys
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None

    # App
    app_name: str = "ARGUS"
    app_version: str = "0.4.0"
    debug: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_enabled: bool = True

    # Response cache
    cache_enabled: bool = True
    cache_ttl_s: int = 3600
    cache_max_bytes: int = 262144

    # Rate limiting (per client)
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 60
    rate_limit_window_s: int = 60
    rate_limit_algorithm: str = "fixed"  # fixed | sliding

    # Deep-report pipeline
    report_max_repair_rounds: int = 1

    # Audio transcription (Sarvam STT)
    sarvam_api_key: str | None = None
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_mode: str = "transcribe"
    sarvam_stt_language: str = "unknown"
    transcription_enabled: bool = True
    transcription_timeout_s: int = 60
    audio_max_upload_bytes: int = 10 * 1024 * 1024

    # Speech output (Sarvam Bulbul TTS)
    speech_enabled: bool = True
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "shubh"
    sarvam_tts_language: str = "en-IN"
    speech_timeout_s: int = 60
    speech_max_chars: int = 1500

    # Conversation memory (working memory, Redis-backed)
    memory_enabled: bool = True
    memory_max_turns: int = 12
    memory_inject_turns: int = 4
    memory_ttl_s: int = 86400
    # Injection budget in tokens (approximated as x4 characters internally)
    memory_token_budget: int = 3000
    # Answers longer than this are stored truncated in history
    memory_max_answer_chars: int = 1200

    # Tracing (opt-in OpenTelemetry)
    tracing_enabled: bool = False
    tracing_exporter: str = "console"
    tracing_otlp_endpoint: str | None = None

    # Authentication (opt-in)
    auth_enabled: bool = False
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    auth_client_id: str | None = None
    auth_client_secret: str | None = None

    # Connector defaults
    connector_timeout_s: int = 45
    connector_max_retries: int = 1
    gemini_model: str = "gemini-3.7-flash"
    mistral_model: str = "mistral-medium-latest"
    # Direct path tries the next provider in the chain when one fails
    direct_failover: bool = True

    # Decomposer
    short_circuit_token_threshold: int = 50
    synthesis_prompt_path: str = "prompts/synthesis_v1.txt"

    # Routing / role binding
    routing_config_path: str = "config/routing.yaml"
    router_strategy: str = "static"
    # A/B experiments: e.g. "semantic:80,static:20". Empty = disabled.
    # Applies only when the caller did NOT set model_config.router_strategy;
    # requests are assigned deterministically by hashing the query text.
    router_ab_split: str = ""

    # Semantic router embeddings ('auto' prefers Gemini when its key exists)
    router_embedding_provider: str = "auto"
    router_embedding_model: str | None = None
    router_embedding_threshold: float = 0.35


settings = Settings()
