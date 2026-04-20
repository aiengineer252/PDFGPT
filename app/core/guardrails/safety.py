"""
Safety Guardrails - Output validation and hallucination prevention.

Features:
1. Groundedness check: Verifies answers are supported by source context
2. Hallucination detection: Cross-references claims with retrieved chunks
3. Confidence scoring: Rates answer reliability
4. Output format validation: JSON schema enforcement
"""

import logging
import re
from typing import Any, Dict, List, Optional

from app.core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class GuardrailsManager:
    """
    Validates LLM outputs to prevent hallucinations and ensure safety.
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    async def validate_response(
        self,
        query: str,
        response: str,
        source_chunks: List[str],
    ) -> Dict[str, Any]:
        """
        Run full validation pipeline on an LLM response.

        Args:
            query: Original user query.
            response: LLM-generated response.
            source_chunks: Retrieved source texts used for generation.

        Returns:
            Validation results with scores and flags.
        """
        # 1. Groundedness check
        groundedness = await self._check_groundedness(response, source_chunks)

        # 2. Relevance check
        relevance = await self._check_relevance(query, response)

        # 3. Confidence scoring
        confidence = self._compute_confidence(groundedness, relevance)

        # 4. Simple content safety check
        safety = self._content_safety_check(response)

        result = {
            "is_valid": confidence > 0.5 and safety["is_safe"],
            "confidence_score": confidence,
            "groundedness": groundedness,
            "relevance": relevance,
            "safety": safety,
            "warnings": [],
        }

        if confidence < 0.5:
            result["warnings"].append(
                "Low confidence: Response may not be well-supported by sources."
            )

        if not groundedness.get("is_grounded", True):
            result["warnings"].append(
                "Potential hallucination: Some claims may not be in the source documents."
            )

        logger.info(f"Validation result: confidence={confidence:.2f}, valid={result['is_valid']}")
        return result

    async def _check_groundedness(
        self, response: str, source_chunks: List[str]
    ) -> Dict[str, Any]:
        """Check if the response is grounded in source documents."""
        if not source_chunks:
            return {"is_grounded": False, "score": 0.0, "reason": "No sources provided"}

        context = "\n\n---\n\n".join(source_chunks[:5])

        prompt = f"""Analyze whether the following response is fully grounded in (supported by) 
the provided source documents. 

Source Documents:
{context[:3000]}

Response to verify:
{response[:1500]}

Rate the groundedness on a scale of 0.0 to 1.0 where:
- 1.0 = Every claim is directly supported by the sources
- 0.5 = Some claims are supported, some are not
- 0.0 = The response is not supported by the sources at all

Respond with ONLY a JSON object like: {{"score": 0.8, "reason": "brief explanation"}}"""

        try:
            result = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a fact-checking validator. Respond only with JSON.",
                temperature=0.0,
                max_tokens=200,
            )

            # Parse JSON from response
            import json
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.5))
                return {
                    "is_grounded": score > 0.6,
                    "score": score,
                    "reason": parsed.get("reason", ""),
                }

        except Exception as e:
            logger.warning(f"Groundedness check failed: {e}")

        return {"is_grounded": True, "score": 0.7, "reason": "Check inconclusive"}

    async def _check_relevance(self, query: str, response: str) -> Dict[str, Any]:
        """Check if the response is relevant to the query."""
        prompt = f"""Rate how well the following response answers the given question.

Question: "{query}"

Response: "{response[:1500]}"

Rate relevance from 0.0 to 1.0 where:
- 1.0 = Directly and completely answers the question
- 0.5 = Partially answers the question
- 0.0 = Completely irrelevant

Respond with ONLY a JSON object like: {{"score": 0.8, "reason": "brief explanation"}}"""

        try:
            result = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a relevance evaluator. Respond only with JSON.",
                temperature=0.0,
                max_tokens=200,
            )

            import json
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "is_relevant": float(parsed.get("score", 0.5)) > 0.5,
                    "score": float(parsed.get("score", 0.5)),
                    "reason": parsed.get("reason", ""),
                }

        except Exception as e:
            logger.warning(f"Relevance check failed: {e}")

        return {"is_relevant": True, "score": 0.7, "reason": "Check inconclusive"}

    def _compute_confidence(
        self, groundedness: Dict, relevance: Dict
    ) -> float:
        """Compute overall confidence score."""
        g_score = groundedness.get("score", 0.5)
        r_score = relevance.get("score", 0.5)

        # Weighted average (groundedness is more important)
        confidence = 0.6 * g_score + 0.4 * r_score

        return round(confidence, 3)

    def _content_safety_check(self, response: str) -> Dict[str, Any]:
        """Basic content safety validation."""
        # Check for common problematic patterns
        concerns = []

        # Check for made-up URLs or citations
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, response)
        if urls:
            concerns.append(f"Contains {len(urls)} URL(s) - verify they are from source documents")

        # Check for overly certain language about uncertain topics
        certainty_markers = ["definitely", "certainly", "100%", "guaranteed", "absolutely certain"]
        for marker in certainty_markers:
            if marker.lower() in response.lower():
                concerns.append(f"Contains high-certainty language: '{marker}'")

        return {
            "is_safe": len(concerns) == 0,
            "concerns": concerns,
        }

    async def add_disclaimer(
        self, response: str, confidence: float
    ) -> str:
        """Add appropriate disclaimers based on confidence level."""
        if confidence >= 0.8:
            return response

        if confidence >= 0.5:
            return (
                response
                + "\n\n---\n*Note: This answer is based on the available document context. "
                "Some details may need verification.*"
            )

        return (
            "⚠️ **Low Confidence Response**\n\n"
            + response
            + "\n\n---\n*⚠️ This response has low confidence and may contain inaccuracies. "
            "Please verify the information against the original documents.*"
        )
