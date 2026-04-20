"""
Dependency Injection - Shared component instances for the API.

Provides singleton instances of all RAG components to route handlers.
"""

import logging
from functools import lru_cache

from app.config import settings
from app.core.llm.llm_manager import LLMManager
from app.core.embeddings.embedding_manager import EmbeddingManager
from app.core.vectorstore.faiss_store import FAISSStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.query_rewriter import QueryRewriter
from app.core.retrieval.reranker import Reranker
from app.core.compression.context_compressor import ContextCompressor
from app.core.memory.conversation_memory import ConversationMemory
from app.core.guardrails.safety import GuardrailsManager
from app.core.evaluation.retrieval_eval import RetrievalEvaluator
from app.core.agent.rag_agent import RAGAgent
from app.pipeline.ingestion import IngestionPipeline
from app.pipeline.query_pipeline import QueryPipeline

logger = logging.getLogger(__name__)

# ── Document Registry ─────────────────────────────────────────
# In-memory registry of uploaded documents
_document_registry: dict = {}


def get_document_registry() -> dict:
    return _document_registry


# ── Singletons ────────────────────────────────────────────────


@lru_cache()
def get_llm_manager() -> LLMManager:
    return LLMManager()


@lru_cache()
def get_embedding_manager() -> EmbeddingManager:
    return EmbeddingManager()


@lru_cache()
def get_faiss_store() -> FAISSStore:
    em = get_embedding_manager()
    return FAISSStore(collection_name="main", dimension=em.embedding_dimension)


@lru_cache()
def get_reranker() -> Reranker:
    return Reranker()


@lru_cache()
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        faiss_store=get_faiss_store(),
        embedding_manager=get_embedding_manager(),
    )


@lru_cache()
def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter(llm_manager=get_llm_manager())


@lru_cache()
def get_compressor() -> ContextCompressor:
    return ContextCompressor(llm_manager=get_llm_manager())


@lru_cache()
def get_memory() -> ConversationMemory:
    return ConversationMemory(llm_manager=get_llm_manager())


@lru_cache()
def get_guardrails() -> GuardrailsManager:
    return GuardrailsManager(llm_manager=get_llm_manager())


@lru_cache()
def get_evaluator() -> RetrievalEvaluator:
    return RetrievalEvaluator(llm_manager=get_llm_manager())


@lru_cache()
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        faiss_store=get_faiss_store(),
        embedding_manager=get_embedding_manager(),
    )


@lru_cache()
def get_query_pipeline() -> QueryPipeline:
    return QueryPipeline(
        llm_manager=get_llm_manager(),
        hybrid_retriever=get_hybrid_retriever(),
        query_rewriter=get_query_rewriter(),
        reranker=get_reranker(),
        compressor=get_compressor(),
        memory=get_memory(),
        guardrails=get_guardrails(),
        evaluator=get_evaluator(),
    )


@lru_cache()
def get_rag_agent() -> RAGAgent:
    return RAGAgent(
        llm_manager=get_llm_manager(),
        hybrid_retriever=get_hybrid_retriever(),
        query_rewriter=get_query_rewriter(),
        reranker=get_reranker(),
        compressor=get_compressor(),
        memory=get_memory(),
    )
