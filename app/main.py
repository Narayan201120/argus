from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.connectors.registry import registry
from app.connectors.gemini import GeminiConnector
from app.connectors.openai import OpenAIConnector
from app.connectors.claude import ClaudeConnector
from app.connectors.mistral import MistralConnector
from app.api.routes import query as query_router
from app.api.routes import health as health_router
from app.api.routes import models as models_router
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info({"message": "ARGUS starting up", "version": settings.app_version})

    registry.register(GeminiConnector())
    registry.register(OpenAIConnector())
    registry.register(ClaudeConnector())
    registry.register(MistralConnector())

    logger.info({"message": "Connectors registered", "connectors": registry.ids()})
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info({"message": "ARGUS shutting down"})


app = FastAPI(
    title="ARGUS",
    description=(
        "Agentic Retrieval & Graph-based Understanding System — "
        "Multimodal AI Orchestration Platform"
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(query_router.router, prefix="/v1", tags=["Query"])
app.include_router(health_router.router, prefix="/v1", tags=["Health"])
app.include_router(models_router.router, prefix="/v1", tags=["Models"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "ARGUS",
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
    }
