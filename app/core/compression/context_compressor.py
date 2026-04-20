"""
Context Compressor - Reduces retrieved context to only relevant information.

Techniques:
1. LLM-based extraction: Pulls only query-relevant sentences
2. Token-aware filtering: Ensures context fits LLM window
3. Redundancy removal: Deduplicates overlapping information
"""

import logging
from typing import Any, Dict, List

from app.core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Compresses retrieved chunks to retain only relevant information.

    This reduces token usage, improves answer quality by removing noise,
    and helps the LLM focus on the most pertinent information.
    """

    def __init__(self, llm_manager: LLMManager, max_context_tokens: int = 4000):
        self.llm = llm_manager
        self.max_context_tokens = max_context_tokens

    async def compress(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compress retrieved documents to only query-relevant content.

        Args:
            query: The user's query.
            documents: Retrieved documents with 'text' field.

        Returns:
            Compressed documents with extracted relevant content.
        """
        if not documents:
            return []

        compressed = []
        total_chars = 0

        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue

            # Skip LLM compression for very short texts
            if len(text) < 200:
                compressed_text = text
            else:
                compressed_text = await self._extract_relevant(query, text)

            # Token-aware cutoff (rough estimate: 1 token ≈ 4 chars)
            if total_chars + len(compressed_text) > self.max_context_tokens * 4:
                # Truncate this chunk to fit
                remaining = (self.max_context_tokens * 4) - total_chars
                if remaining > 100:
                    compressed_text = compressed_text[:remaining]
                else:
                    break

            compressed.append({
                **doc,
                "text": compressed_text,
                "original_text": text,
                "compressed": True,
                "compression_ratio": len(compressed_text) / max(len(text), 1),
            })
            total_chars += len(compressed_text)

        logger.info(
            f"Compressed {len(documents)} documents → {len(compressed)} "
            f"(total chars: {total_chars})"
        )
        return compressed

    async def _extract_relevant(self, query: str, text: str) -> str:
        """Extract only the sentences relevant to the query using LLM."""
        prompt = f"""Given the following query and text passage, extract ONLY the sentences 
that are directly relevant to answering the query. Remove irrelevant information.

Query: "{query}"

Text passage:
{text[:2000]}

Extracted relevant content:"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise information extractor. "
                    "Extract only relevant sentences. Do not add any new information. "
                    "If nothing is relevant, return 'No relevant information found.'"
                ),
                temperature=0.0,
                max_tokens=1024,
            )
            result = response.strip()
            if result and "no relevant information" not in result.lower():
                return result
            return text[:500]  # Fallback to truncation

        except Exception as e:
            logger.warning(f"Context compression failed: {e}")
            return text[:500]

    def remove_redundancy(
        self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate documents based on text overlap.

        Args:
            documents: List of documents.
            similarity_threshold: Jaccard similarity threshold for dedup.

        Returns:
            Deduplicated document list.
        """
        if len(documents) <= 1:
            return documents

        unique = [documents[0]]

        for doc in documents[1:]:
            is_duplicate = False
            doc_words = set(doc["text"].lower().split())

            for existing in unique:
                existing_words = set(existing["text"].lower().split())

                # Jaccard similarity
                intersection = len(doc_words & existing_words)
                union = len(doc_words | existing_words)

                if union > 0 and (intersection / union) > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(doc)

        if len(unique) < len(documents):
            logger.info(
                f"Redundancy removal: {len(documents)} → {len(unique)} documents"
            )

        return unique
