"""
Query Pipeline - Full query processing pipeline.

Pipeline:
1. Conversation context resolution
2. Query rewriting/expansion
3. Hybrid retrieval (vector + BM25)
4. Re-ranking
5. Context compression
6. LLM generation (with streaming support)
7. Guardrails validation
8. Evaluation metrics
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.llm.llm_manager import LLMManager
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.query_rewriter import QueryRewriter
from app.core.retrieval.reranker import Reranker
from app.core.compression.context_compressor import ContextCompressor
from app.core.memory.conversation_memory import ConversationMemory
from app.core.guardrails.safety import GuardrailsManager
from app.core.evaluation.retrieval_eval import RetrievalEvaluator

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Full RAG query processing pipeline with all advanced components.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        hybrid_retriever: HybridRetriever,
        query_rewriter: QueryRewriter,
        reranker: Reranker,
        compressor: ContextCompressor,
        memory: ConversationMemory,
        guardrails: GuardrailsManager,
        evaluator: RetrievalEvaluator,
    ):
        self.llm = llm_manager
        self.retriever = hybrid_retriever
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.compressor = compressor
        self.memory = memory
        self.guardrails = guardrails
        self.evaluator = evaluator

    async def query(
        self,
        question: str,
        session_id: str = "default",
        doc_id_filter: Optional[str] = None,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
        enable_compression: bool = True,
        enable_guardrails: bool = True,
        enable_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full query pipeline.

        Args:
            question: User's question.
            session_id: Conversation session ID.
            doc_id_filter: Optional filter by document ID.
            enable_rewrite: Whether to use query rewriting.
            enable_rerank: Whether to use re-ranking.
            enable_compression: Whether to use context compression.
            enable_guardrails: Whether to validate output.
            enable_evaluation: Whether to compute eval metrics.

        Returns:
            Complete response with answer, sources, scores, and metadata.
        """
        pipeline_metadata = {"steps": []}

        # Step 1: Conversation context resolution
        contextualized = await self.memory.get_contextualized_query(
            session_id, question
        )
        pipeline_metadata["steps"].append(
            f"Contextualized: '{question}' → '{contextualized}'"
        )

        # Step 2: Query rewriting
        search_query = contextualized
        if enable_rewrite:
            search_query = await self.query_rewriter.rewrite_query(
                contextualized,
                self.memory.get_context_window(session_id),
            )
            pipeline_metadata["steps"].append(f"Rewritten: '{search_query}'")

        # Step 3: Hybrid retrieval
        retrieved = self.retriever.retrieve(
            search_query, doc_id_filter=doc_id_filter
        )
        pipeline_metadata["steps"].append(f"Retrieved {len(retrieved)} results")

        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "sources": [],
                "pipeline_metadata": pipeline_metadata,
            }

        # Step 4: Re-ranking
        if enable_rerank:
            reranked = self.reranker.rerank(search_query, retrieved)
            pipeline_metadata["steps"].append(
                f"Re-ranked to {len(reranked)} top results"
            )
        else:
            reranked = retrieved[:5]

        # Step 5: Context compression
        if enable_compression:
            compressed = await self.compressor.compress(search_query, reranked)
            # Remove redundancy
            compressed = self.compressor.remove_redundancy(compressed)
            pipeline_metadata["steps"].append(
                f"Compressed to {len(compressed)} chunks"
            )
        else:
            compressed = reranked

        # Step 6: Generate answer
        context = "\n\n---\n\n".join([doc["text"] for doc in compressed])

        prompt = self._build_answer_prompt(contextualized, context)
        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=(
                "You are a knowledgeable document assistant. Answer questions "
                "based strictly on the provided context. Be thorough, precise, "
                "and cite relevant sections from the documents."
            ),
            temperature=0.1,
            max_tokens=2048,
        )
        answer = answer.strip()
        pipeline_metadata["steps"].append("Answer generated")

        # Step 7: Guardrails validation
        validation = {}
        if enable_guardrails:
            source_texts = [doc["text"] for doc in compressed]
            validation = await self.guardrails.validate_response(
                contextualized, answer, source_texts
            )
            pipeline_metadata["steps"].append(
                f"Validation: confidence={validation.get('confidence_score', 0):.2f}"
            )

            # Add disclaimer if low confidence
            if validation.get("confidence_score", 1.0) < 0.8:
                answer = await self.guardrails.add_disclaimer(
                    answer, validation["confidence_score"]
                )

        # Step 8: Evaluation metrics
        evaluation = {}
        if enable_evaluation:
            contexts = [doc["text"] for doc in compressed]
            evaluation = await self.evaluator.evaluate(
                contextualized, answer, contexts
            )
            pipeline_metadata["steps"].append(
                f"Evaluation: {evaluation.get('grade', 'N/A')} "
                f"(score={evaluation.get('overall_score', 0):.2f})"
            )

        # Store in memory
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", answer)

        # Build sources
        sources = []
        for doc in compressed:
            sources.append({
                "text": doc.get("original_text", doc["text"])[:300],
                "metadata": doc.get("metadata", {}),
                "score": doc.get("rerank_score", doc.get("score", 0)),
                "retrieval_method": doc.get("retrieval_method", "unknown"),
            })

        return {
            "answer": answer,
            "sources": sources,
            "search_query": search_query,
            "validation": validation,
            "evaluation": evaluation,
            "pipeline_metadata": pipeline_metadata,
        }

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        doc_id_filter: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the answer token by token.

        Performs retrieval, reranking, and compression first,
        then streams the LLM generation.
        """
        # Pre-processing (non-streaming)
        contextualized = await self.memory.get_contextualized_query(
            session_id, question
        )
        search_query = await self.query_rewriter.rewrite_query(
            contextualized,
            self.memory.get_context_window(session_id),
        )

        retrieved = self.retriever.retrieve(search_query, doc_id_filter=doc_id_filter)

        if not retrieved:
            yield "I couldn't find relevant information in the documents to answer your question."
            return

        reranked = self.reranker.rerank(search_query, retrieved)
        compressed = await self.compressor.compress(search_query, reranked)

        context = "\n\n---\n\n".join([doc["text"] for doc in compressed])
        prompt = self._build_answer_prompt(contextualized, context)

        # Stream the answer
        full_answer = ""
        async for token in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=(
                "You are a knowledgeable document assistant. Answer questions "
                "based strictly on the provided context."
            ),
            temperature=0.1,
            max_tokens=2048,
        ):
            full_answer += token
            yield token

        # Store in memory after streaming completes
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", full_answer)

    def _build_answer_prompt(self, question: str, context: str) -> str:
        """Build the prompt for answer generation."""
        return f"""Based on the following context extracted from the uploaded documents, 
answer the user's question thoroughly and accurately.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer ONLY based on the provided context
- If the context doesn't contain enough information to fully answer, clearly state what information is missing
- Reference specific parts of the documents when applicable
- Be detailed and well-organized in your response
- Use bullet points or numbered lists for clarity when appropriate

ANSWER:"""
