"""
RAG Agent - Agentic RAG with dynamic tool selection.

Instead of a fixed pipeline, the agent intelligently decides:
- When to retrieve vs answer directly
- Whether to rewrite the query first
- Which retrieval strategy to use
- Whether to do additional retrieval rounds
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.llm.llm_manager import LLMManager
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.query_rewriter import QueryRewriter
from app.core.retrieval.reranker import Reranker
from app.core.compression.context_compressor import ContextCompressor
from app.core.memory.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)


class AgentAction(str, Enum):
    """Actions the agent can take."""
    RETRIEVE = "retrieve"
    REWRITE_AND_RETRIEVE = "rewrite_and_retrieve"
    ANSWER_DIRECTLY = "answer_directly"
    CLARIFY = "clarify"
    SUMMARIZE = "summarize"


class RAGAgent:
    """
    Intelligent agent that orchestrates the RAG pipeline dynamically.

    The agent analyzes each query and decides the optimal strategy,
    rather than always following the same fixed pipeline.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        hybrid_retriever: HybridRetriever,
        query_rewriter: QueryRewriter,
        reranker: Reranker,
        compressor: ContextCompressor,
        memory: ConversationMemory,
    ):
        self.llm = llm_manager
        self.retriever = hybrid_retriever
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.compressor = compressor
        self.memory = memory

    async def process_query(
        self,
        query: str,
        session_id: str = "default",
        doc_id_filter: Optional[str] = None,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Process a query through the agentic RAG pipeline.

        The agent decides the best strategy for each query.

        Args:
            query: User query.
            session_id: Session ID for conversation memory.
            doc_id_filter: Optional document ID filter.
            max_iterations: Maximum retrieval rounds.

        Returns:
            Response dict with answer, sources, and agent reasoning.
        """
        # Step 1: Contextualize query using conversation history
        contextualized_query = await self.memory.get_contextualized_query(
            session_id, query
        )

        # Step 2: Decide action
        action = await self._decide_action(contextualized_query, session_id)
        logger.info(f"Agent decided: {action.value} for query: '{contextualized_query}'")

        # Step 3: Execute the decided action
        reasoning_steps = [f"Action: {action.value}"]
        result = None

        if action == AgentAction.ANSWER_DIRECTLY:
            result = await self._answer_directly(contextualized_query)
            reasoning_steps.append("Answered directly without retrieval")

        elif action == AgentAction.CLARIFY:
            result = {
                "answer": "I need more information to answer your question accurately. Could you please provide more details or context?",
                "sources": [],
                "needs_clarification": True,
            }
            reasoning_steps.append("Requested clarification from user")

        elif action == AgentAction.REWRITE_AND_RETRIEVE:
            # Rewrite then retrieve
            rewritten = await self.query_rewriter.rewrite_query(
                contextualized_query,
                self.memory.get_context_window(session_id),
            )
            reasoning_steps.append(f"Rewrote query: '{rewritten}'")
            result = await self._retrieve_and_answer(
                rewritten, doc_id_filter, max_iterations, reasoning_steps
            )

        elif action == AgentAction.SUMMARIZE:
            result = await self._retrieve_and_summarize(
                contextualized_query, doc_id_filter
            )
            reasoning_steps.append("Generated summary from retrieved content")

        else:  # RETRIEVE
            result = await self._retrieve_and_answer(
                contextualized_query, doc_id_filter, max_iterations, reasoning_steps
            )

        # Step 4: Store in memory
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(
            session_id, "assistant", result.get("answer", "")
        )

        result["reasoning"] = reasoning_steps
        result["action_taken"] = action.value
        return result

    async def _decide_action(self, query: str, session_id: str) -> AgentAction:
        """Use LLM to decide the best action for this query."""
        prompt = f"""Analyze the following query and decide the best action to take.

Query: "{query}"

Available actions:
1. "retrieve" - Search documents and answer based on retrieved content
2. "rewrite_and_retrieve" - Improve the query first, then search (use for vague/ambiguous queries)
3. "answer_directly" - Answer without searching (for greetings, clarifications, or general knowledge)
4. "clarify" - Ask user for more information (query is too vague to search)
5. "summarize" - Retrieve and provide a comprehensive summary

Respond with ONLY one of: retrieve, rewrite_and_retrieve, answer_directly, clarify, summarize"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a routing agent. Respond with only the action name.",
                temperature=0.0,
                max_tokens=20,
            )

            action_str = response.strip().lower().replace('"', '').replace("'", "")

            action_map = {
                "retrieve": AgentAction.RETRIEVE,
                "rewrite_and_retrieve": AgentAction.REWRITE_AND_RETRIEVE,
                "answer_directly": AgentAction.ANSWER_DIRECTLY,
                "clarify": AgentAction.CLARIFY,
                "summarize": AgentAction.SUMMARIZE,
            }

            return action_map.get(action_str, AgentAction.RETRIEVE)

        except Exception as e:
            logger.warning(f"Action decision failed: {e}. Defaulting to RETRIEVE.")
            return AgentAction.RETRIEVE

    async def _retrieve_and_answer(
        self,
        query: str,
        doc_id_filter: Optional[str],
        max_iterations: int,
        reasoning_steps: List[str],
    ) -> Dict[str, Any]:
        """Standard retrieve → rerank → compress → answer pipeline."""

        # Retrieve
        results = self.retriever.retrieve(query, doc_id_filter=doc_id_filter)
        reasoning_steps.append(f"Retrieved {len(results)} candidates")

        if not results:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents to answer your question.",
                "sources": [],
            }

        # Rerank
        reranked = self.reranker.rerank(query, results)
        reasoning_steps.append(f"Re-ranked to top {len(reranked)} results")

        # Check if results are good enough
        best_score = reranked[0].get("rerank_score", 0) if reranked else 0

        # If best score is low, try query expansion for a second round
        if best_score < -2.0 and max_iterations > 1:
            expanded_queries = await self.query_rewriter.expand_query(query, 2)
            reasoning_steps.append(f"Low relevance ({best_score:.2f}), expanding query")

            for eq in expanded_queries[1:]:  # Skip original
                extra_results = self.retriever.retrieve(eq, doc_id_filter=doc_id_filter)
                results.extend(extra_results)

            # Re-rank the combined results
            reranked = self.reranker.rerank(query, results)
            reasoning_steps.append(
                f"Re-ranked expanded results: {len(reranked)} final candidates"
            )

        # Compress
        compressed = await self.compressor.compress(query, reranked)
        reasoning_steps.append(f"Compressed to {len(compressed)} relevant chunks")

        # Generate answer
        context = "\n\n---\n\n".join(
            [doc["text"] for doc in compressed]
        )

        prompt = f"""Based on the following context from the uploaded documents, answer the question.

Context:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite relevant sections when possible
- Be detailed and thorough

Answer:"""

        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt=(
                "You are a knowledgeable assistant that answers questions "
                "based strictly on the provided document context. Be accurate and thorough."
            ),
            temperature=0.1,
            max_tokens=2048,
        )

        # Collect sources
        sources = []
        for doc in compressed:
            sources.append({
                "text": doc.get("original_text", doc["text"])[:200],
                "metadata": doc.get("metadata", {}),
                "score": doc.get("rerank_score", doc.get("score", 0)),
            })

        return {
            "answer": answer.strip(),
            "sources": sources,
        }

    async def _answer_directly(self, query: str) -> Dict[str, Any]:
        """Answer without retrieval (for simple/general queries)."""
        answer = await self.llm.generate(
            prompt=query,
            system_prompt=(
                "You are a helpful assistant. Answer the query directly. "
                "If it seems like a question about a specific document, "
                "let the user know they should ask about specific document content."
            ),
            temperature=0.3,
            max_tokens=512,
        )

        return {
            "answer": answer.strip(),
            "sources": [],
        }

    async def _retrieve_and_summarize(
        self, query: str, doc_id_filter: Optional[str]
    ) -> Dict[str, Any]:
        """Retrieve broadly and provide a comprehensive summary."""
        results = self.retriever.retrieve(query, top_k=30, doc_id_filter=doc_id_filter)

        if not results:
            return {
                "answer": "No relevant content found to summarize.",
                "sources": [],
            }

        # Take more results for summarization
        context = "\n\n---\n\n".join([r["text"] for r in results[:15]])

        prompt = f"""Based on the following content from the documents, provide a comprehensive summary 
related to the query.

Query: {query}

Content:
{context[:6000]}

Provide a well-structured, detailed summary:"""

        answer = await self.llm.generate(
            prompt=prompt,
            system_prompt="You are a document summarizer. Provide thorough, well-organized summaries.",
            temperature=0.2,
            max_tokens=2048,
        )

        sources = [
            {
                "text": r["text"][:200],
                "metadata": r.get("metadata", {}),
                "score": r.get("score", 0),
            }
            for r in results[:5]
        ]

        return {
            "answer": answer.strip(),
            "sources": sources,
        }
