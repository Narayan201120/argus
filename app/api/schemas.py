from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class ConnectorConfigRequest(BaseModel):
    connectors: list[str] = Field(
        default=["gemini", "openai", "claude"],
        description="Connector IDs to activate for this query",
    )
    timeout_s: int = Field(default=45, ge=5, le=120)
    max_tokens: int = Field(default=4096, ge=256, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=32000)
    model_config_: ConnectorConfigRequest = Field(
        default_factory=ConnectorConfigRequest,
        alias="model_config",
    )

    model_config = {"populate_by_name": True}


class TokenUsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelStatus(BaseModel):
    connector_id: str
    status: Literal["success", "timeout", "error", "skipped"]
    latency_ms: int
    error: Optional[str] = None
    token_usage: Optional[TokenUsageOut] = None
    sub_query: Optional[str] = None


class QueryResponse(BaseModel):
    request_id: str
    query: str
    result: str
    synthesizer: str
    model_statuses: list[ModelStatus]
    latency_breakdown: dict[str, int]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    short_circuited: bool = False


class ConnectorProfile(BaseModel):
    connector_id: str
    display_name: str
    capabilities: list[str]
    is_available: bool


class ModelsResponse(BaseModel):
    connectors: list[ConnectorProfile]
    total: int


class ConnectorHealthStatus(BaseModel):
    connector_id: str
    is_available: bool
    status: Literal["ok", "degraded", "unavailable"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    connectors: list[ConnectorHealthStatus]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
