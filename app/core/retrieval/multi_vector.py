"""
Multi-Vector Retriever - Multiple representations per document chunk.

Creates both raw chunk embeddings AND summary embeddings for each chunk,
enabling retrieval via summaries while returning full chunk content.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.embeddings.embedding_manager import EmbeddingManager
from app.core.llm.llm_manager import LLMManager
from app.core.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class MultiVectorRetriever:
    """
    Stores and retrieves from multiple vector representations per chunk.

    Strategy:
    1. Raw chunk embedding → for detailed matching
    2. Summary embedding → for conceptual/high-level matching

    Retrieval returns the full original chunk regardless of which
    representation matched.
    """

    def __init__(
        self,
        raw_store: FAISSStore,
        summary_store: FAISSStore,
        embedding_manager: EmbeddingManager,
        llm_manager: LLMManager,
    ):
        self.raw_store = raw_store
        self.summary_store = summary_store
        self.embedding_manager = embedding_manager
        self.llm = llm_manager

    async def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        doc_id: str,
    ) -> int:
        """
        Add chunks with both raw and summary representations.

        Args:
            chunks: List of {text, metadata} dicts.
            doc_id: Document identifier.

        Returns:
            Number of chunks processed.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadata_list = [c.get("metadata", {}) for c in chunks]

        # 1. Generate and store raw embeddings
        raw_embeddings = self.embedding_manager.embed_documents(texts)
        self.raw_store.add(raw_embeddings, texts, metadata_list, doc_id)

        # 2. Generate summaries and store summary embeddings
        summaries = await self._generate_summaries(texts)
        summary_embeddings = self.embedding_manager.embed_documents(summaries)

        # For summary store, we store the SUMMARY as searchable text
        # but keep original text in metadata for retrieval
        summary_metadata = []
        for i, meta in enumerate(metadata_list):
            summary_metadata.append({
                **meta,
                "original_text": texts[i],
                "summary": summaries[i],
                "representation": "summary",
            })

        self.summary_store.add(summary_embeddings, summaries, summary_metadata, doc_id)

        logger.info(f"Multi-vector: added {len(texts)} chunks with raw + summary representations")
        return len(texts)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using both raw and summary representations.

        Args:
            query: Search query.
            top_k: Number of results per representation.
            doc_id_filter: Optional document filter.

        Returns:
            Deduplicated results from both representations.
        """
        query_embedding = self.embedding_manager.embed_query(query)

        # Search raw embeddings
        raw_results = self.raw_store.search(query_embedding, top_k, doc_id_filter)
        for r in raw_results:
            r["representation"] = "raw"

        # Search summary embeddings
        summary_results = self.summary_store.search(query_embedding, top_k, doc_id_filter)
        for r in summary_results:
            # Replace summary text with original text for downstream use
            if "original_text" in r.get("metadata", {}):
                r["text"] = r["metadata"]["original_text"]
            r["representation"] = "summary"

        # Deduplicate and merge
        seen_texts = set()
        merged = []

        # Interleave results (raw first for each rank position)
        for results in [raw_results, summary_results]:
            for result in results:
                text_key = result["text"][:100]  # Use first 100 chars as key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    merged.append(result)

        # Sort by score and return top_k
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:top_k]

    async def _generate_summaries(self, texts: List[str]) -> List[str]:
        """Generate concise summaries for each chunk using LLM."""
        summaries = []

        for text in texts:
            try:
                prompt = f"""Summarize the following text in 1-2 sentences. 
Capture the key concepts, entities, and main ideas.

Text: {text[:1500]}

Summary:"""

                summary = await self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are a concise summarizer. Write brief, informative summaries.",
                    temperature=0.1,
                    max_tokens=128,
                )
                summaries.append(summary.strip())
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}. Using truncated text.")
                summaries.append(text[:200])

        return summaries
