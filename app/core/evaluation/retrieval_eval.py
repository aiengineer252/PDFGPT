"""
Retrieval Evaluator - RAGAS-style evaluation metrics.

Measures:
1. Context Relevance: How relevant are retrieved chunks to the query
2. Faithfulness: Is the answer supported by the context
3. Answer Relevance: Does the answer address the query
4. Overall quality score
"""

import logging
import re
from typing import Any, Dict, List

from app.core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    RAGAS-style evaluation for RAG pipeline quality.

    Evaluates three key dimensions:
    1. Context Relevance (retrieval quality)
    2. Faithfulness (generation grounding)
    3. Answer Relevance (response quality)
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Run full RAGAS-style evaluation.

        Args:
            query: Original query.
            answer: Generated answer.
            contexts: Retrieved context chunks.

        Returns:
            Evaluation metrics dict.
        """
        # Run evaluations (could be parallelized with asyncio.gather)
        context_relevance = await self._evaluate_context_relevance(query, contexts)
        faithfulness = await self._evaluate_faithfulness(answer, contexts)
        answer_relevance = await self._evaluate_answer_relevance(query, answer)

        # Compute overall score
        overall = (
            context_relevance["score"] * 0.3
            + faithfulness["score"] * 0.4
            + answer_relevance["score"] * 0.3
        )

        result = {
            "overall_score": round(overall, 3),
            "context_relevance": context_relevance,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "num_contexts": len(contexts),
            "grade": self._score_to_grade(overall),
        }

        logger.info(
            f"Evaluation: overall={overall:.2f}, "
            f"context={context_relevance['score']:.2f}, "
            f"faithful={faithfulness['score']:.2f}, "
            f"relevant={answer_relevance['score']:.2f}"
        )

        return result

    async def _evaluate_context_relevance(
        self, query: str, contexts: List[str]
    ) -> Dict[str, Any]:
        """Evaluate how relevant the retrieved contexts are to the query."""
        if not contexts:
            return {"score": 0.0, "reason": "No contexts retrieved"}

        context_text = "\n\n---\n\n".join(contexts[:5])

        prompt = f"""Evaluate how relevant the following retrieved contexts are to the query.

Query: "{query}"

Retrieved Contexts:
{context_text[:3000]}

Rate context relevance from 0.0 to 1.0:
- 1.0 = All contexts are highly relevant to answering the query
- 0.5 = Some contexts are relevant, others are not
- 0.0 = No context is relevant to the query

Respond with ONLY a JSON object: {{"score": 0.8, "reason": "brief explanation"}}"""

        return await self._llm_evaluate(prompt)

    async def _evaluate_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> Dict[str, Any]:
        """Evaluate if the answer is faithful to (supported by) the contexts."""
        if not contexts:
            return {"score": 0.0, "reason": "No source context available"}

        context_text = "\n\n---\n\n".join(contexts[:5])

        prompt = f"""Evaluate whether each claim in the answer is supported by the source contexts.

Source Contexts:
{context_text[:3000]}

Answer to evaluate:
{answer[:1500]}

Rate faithfulness from 0.0 to 1.0:
- 1.0 = Every statement in the answer is directly supported by the contexts
- 0.5 = Some statements are supported, some are made up
- 0.0 = The answer contains mostly unsupported claims

Respond with ONLY a JSON object: {{"score": 0.8, "reason": "brief explanation"}}"""

        return await self._llm_evaluate(prompt)

    async def _evaluate_answer_relevance(
        self, query: str, answer: str
    ) -> Dict[str, Any]:
        """Evaluate if the answer is relevant to and addresses the query."""
        prompt = f"""Evaluate how well the answer addresses the given query.

Query: "{query}"

Answer: "{answer[:1500]}"

Rate answer relevance from 0.0 to 1.0:
- 1.0 = The answer directly and completely addresses the query
- 0.5 = The answer partially addresses the query
- 0.0 = The answer does not address the query at all

Respond with ONLY a JSON object: {{"score": 0.8, "reason": "brief explanation"}}"""

        return await self._llm_evaluate(prompt)

    async def _llm_evaluate(self, prompt: str) -> Dict[str, Any]:
        """Helper to run LLM evaluation and parse JSON response."""
        try:
            result = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are an evaluation expert. Respond only with JSON.",
                temperature=0.0,
                max_tokens=200,
            )

            import json
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "score": float(parsed.get("score", 0.5)),
                    "reason": parsed.get("reason", ""),
                }

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

        return {"score": 0.5, "reason": "Evaluation inconclusive"}

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"
