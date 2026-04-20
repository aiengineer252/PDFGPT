"""
Re-ranker - Cross-encoder model for precise result re-ranking.

After initial retrieval, uses a cross-encoder to re-score
query-document pairs for much higher precision top-K results.
"""

import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder re-ranker for precision improvement.

    Unlike bi-encoder (embedding) search which encodes query and document separately,
    cross-encoders process the query-document pair together, enabling much more
    accurate relevance scoring at the cost of speed (hence used as a second stage).
    """

    _instance: Optional["Reranker"] = None
    _model: Optional[CrossEncoder] = None

    def __new__(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
        self._model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = settings.TOP_K_RERANK,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved documents using cross-encoder scoring.

        Args:
            query: The search query.
            documents: List of retrieved documents with 'text' field.
            top_k: Number of top results to return after reranking.

        Returns:
            Re-ranked list of documents with updated scores.
        """
        if not documents:
            return []

        if len(documents) <= 1:
            return documents

        # Prepare query-document pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Score all pairs
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach reranker scores
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            doc["original_score"] = doc.get("score", 0.0)

        # Sort by reranker score (descending)
        reranked = sorted(
            documents,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        logger.info(
            f"Re-ranked {len(documents)} documents → top {top_k}. "
            f"Best score: {reranked[0]['rerank_score']:.4f}"
        )

        return reranked[:top_k]
