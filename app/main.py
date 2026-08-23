from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import audio as audio_router
from app.api.routes import auth as auth_router
from app.api.routes import health as health_router
from app.api.routes import models as models_router
from app.api.routes import query as query_router
from app.api.routes import reports as reports_router
from app.api.routes import stream as stream_router
from app.auth import JWTAuthMiddleware
from app.cache import ResponseCache
from app.config import settings
from app.connectors.claude import ClaudeConnector
from app.connectors.gemini import GeminiConnector
from app.connectors.mistral import MistralConnector
from app.connectors.openai import OpenAIConnector
from app.connectors.registry import registry
from app.metrics import PrometheusMiddleware
from app.metrics import router as metrics_router
from app.ratelimit import RateLimitMiddleware
from app.rediskit import close_redis, connect_redis, holder
from app.tracing import configure_tracing
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info({"message": "ARGUS starting up", "version": settings.app_version})

    configure_tracing()

    registry.register(GeminiConnector())
    registry.register(OpenAIConnector())
    registry.register(ClaudeConnector())
    registry.register(MistralConnector())

    logger.info({"message": "Connectors registered", "connectors": registry.ids()})

    holder.client = await connect_redis()
    holder.cache = ResponseCache(holder.client) if holder.client else None

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info({"message": "ARGUS shutting down"})
    holder.cache = None
    await close_redis()


app = FastAPI(
    title="ARGUS",
    description=(
        "Agentic Retrieval & Graph-based Understanding System — "
        "Multimodal AI Orchestration Platform"
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware, holder=holder)
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(PrometheusMiddleware)

app.include_router(auth_router.router, prefix="/v1", tags=["Auth"])
app.include_router(audio_router.router, prefix="/v1", tags=["Audio"])
app.include_router(query_router.router, prefix="/v1", tags=["Query"])
app.include_router(stream_router.router, prefix="/v1", tags=["Query"])
app.include_router(reports_router.router, prefix="/v1", tags=["Reports"])
app.include_router(health_router.router, prefix="/v1", tags=["Health"])
app.include_router(models_router.router, prefix="/v1", tags=["Models"])
app.include_router(metrics_router, prefix="/v1", tags=["Metrics"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "ARGUS",
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
    }
