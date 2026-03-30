"""FastAPI application entry point for the Orchestrator Agent."""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes import agents, analyze, datasets, plan, stream
from app.session_store import close_redis

# ── Logging ───────────────────────────────────────────────────────────────────

# Force UTF-8 on Windows so emoji in 3rd-party libs don't crash the logger.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "[START] Orchestrator Agent starting -- provider=%s  port=%d  ec2=%s",
        settings.llm_provider, settings.port, settings.use_ec2,
    )
    yield
    logger.info("[STOP] Orchestrator shutting down -- closing Redis connection")
    await close_redis()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Orchestrator Agent",
        description=(
            "Central planner for the GEMRSLIZE multi-agent data analysis platform. "
            "Accepts natural-language queries, decomposes them into a task graph via an LLM, "
            "dispatches to specialist agents, and streams or returns unified results."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — configure CORS_ORIGINS env var for production (comma-separated)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(analyze.router)
    app.include_router(stream.router)
    app.include_router(plan.router)
    app.include_router(agents.router)
    app.include_router(datasets.router)

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["health"], summary="Service liveness probe")
    async def health():
        s = get_settings()
        provider = str(getattr(s, "llm_provider", "unknown"))
        model = ""
        if provider == "azure_openai":
            model = getattr(s, "azure_openai_deployment_name", "")
        elif provider == "groq":
            model = getattr(s, "groq_model", "")
        elif provider == "openai":
            model = "gpt-4o"
        elif provider == "ollama":
            model = getattr(s, "ollama_model", "")
            
        return {
            "status": "ok", 
            "service": "orchestrator-agent", 
            "llm_provider": provider,
            "llm_model": model
        }

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
