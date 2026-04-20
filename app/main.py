"""
FastAPI Application - Main entry point.

Features:
- CORS middleware for frontend integration
- Organized route registration
- Startup/shutdown lifecycle events
- Logging configuration
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, query
from app.config import settings

# ── Logging Setup ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ── Application Lifecycle ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 PDFGPT - Advanced RAG System Starting...")
    logger.info(f"   LLM Provider : {settings.LLM_PROVIDER}")
    logger.info(f"   Embedding    : {settings.EMBEDDING_MODEL}")
    logger.info(f"   Reranker     : {settings.RERANKER_MODEL}")
    logger.info(f"   Chunk Strategy: {settings.CHUNKING_STRATEGY}")
    logger.info(f"   Hybrid Alpha : {settings.HYBRID_ALPHA}")
    logger.info("=" * 60)

    # Pre-load models (warm up)
    logger.info("Loading models (first request will be faster)...")
    try:
        from app.api.deps import get_embedding_manager
        get_embedding_manager()
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.warning(f"⚠️  Embedding model pre-load failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down PDFGPT...")
    try:
        from app.api.deps import get_llm_manager
        llm = get_llm_manager()
        await llm.close()
    except Exception:
        pass
    logger.info("👋 PDFGPT shutdown complete")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="PDFGPT - Advanced RAG System",
    description=(
        "Production-grade Advanced RAG system for PDF querying. "
        "Features: Smart Chunking, Hybrid Retrieval (Vector + BM25), "
        "Re-ranking, Context Compression, Query Rewriting, "
        "Agentic RAG, Guardrails, and RAGAS Evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")


# ── Root Endpoint ─────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with system info."""
    return {
        "name": "PDFGPT - Advanced RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "features": [
            "Smart Chunking (Semantic, Recursive, Context-Aware)",
            "Hybrid Retrieval (Vector + BM25 with RRF)",
            "Cross-Encoder Re-ranking",
            "LLM Query Rewriting & HyDE",
            "Multi-Vector Retrieval",
            "Context Compression",
            "Conversation Memory",
            "Guardrails & Hallucination Detection",
            "RAGAS-Style Evaluation",
            "Agentic RAG",
            "Streaming Responses (SSE)",
        ],
    }
