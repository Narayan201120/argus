from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # AI Provider API Keys
    gemini_api_key: Optional[str] = None
    # openai_api_key: Optional[str] = None
    # anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None

    # App
    app_name: str = "ARGUS"
    app_version: str = "0.1.0"
    debug: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Connector defaults
    connector_timeout_s: int = 45
    connector_max_retries: int = 1

    # Decomposer
    short_circuit_token_threshold: int = 50
    synthesis_prompt_path: str = "prompts/synthesis_v1.txt"


settings = Settings()
