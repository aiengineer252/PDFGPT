"""
Health Check Route - System status and diagnostics.
"""

from fastapi import APIRouter

from app.api.deps import get_faiss_store
from app.api.schemas import HealthResponse
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and return configuration info."""
    store = get_faiss_store()
    stats = store.get_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL,
        total_vectors=stats.get("total_vectors", 0),
        total_documents=stats.get("unique_documents", 0),
    )
