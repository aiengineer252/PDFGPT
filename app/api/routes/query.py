"""
Query Routes - RAG query, streaming, agentic, and evaluation endpoints.
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import (
    get_evaluator,
    get_memory,
    get_query_pipeline,
    get_rag_agent,
)
from app.api.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    ConversationHistoryResponse,
    EvaluationRequest,
    EvaluationResponse,
    QueryRequest,
    QueryResponse,
    SessionInfo,
    SourceInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with the full pipeline.

    Pipeline stages:
    1. Conversation context resolution
    2. Query rewriting
    3. Hybrid retrieval (vector + BM25)
    4. Cross-encoder re-ranking
    5. Context compression
    6. LLM answer generation
    7. Guardrails validation
    8. (Optional) RAGAS evaluation
    """
    try:
        pipeline = get_query_pipeline()
        result = await pipeline.query(
            question=request.question,
            session_id=request.session_id,
            doc_id_filter=request.doc_id,
            enable_rewrite=request.enable_rewrite,
            enable_rerank=request.enable_rerank,
            enable_compression=request.enable_compression,
            enable_guardrails=request.enable_guardrails,
            enable_evaluation=request.enable_evaluation,
        )

        return QueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result.get("sources", [])],
            search_query=result.get("search_query"),
            validation=result.get("validation", {}),
            evaluation=result.get("evaluation", {}),
            pipeline_metadata=result.get("pipeline_metadata", {}),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/stream")
async def query_stream(request: QueryRequest):
    """
    Stream the RAG response token by token.

    Uses Server-Sent Events (SSE) format for real-time streaming.
    """
    async def event_generator():
        try:
            pipeline = get_query_pipeline()
            async for token in pipeline.query_stream(
                question=request.question,
                session_id=request.session_id,
                doc_id_filter=request.doc_id,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/agent", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Agentic RAG query - the AI agent decides the optimal strategy.

    The agent can:
    - Retrieve from documents
    - Rewrite the query first
    - Answer directly for greetings/general questions
    - Request clarification for vague queries
    - Generate comprehensive summaries
    """
    try:
        agent = get_rag_agent()
        result = await agent.process_query(
            query=request.question,
            session_id=request.session_id,
            doc_id_filter=request.doc_id,
        )

        return AgentQueryResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result.get("sources", [])],
            action_taken=result.get("action_taken", ""),
            reasoning=result.get("reasoning", []),
            needs_clarification=result.get("needs_clarification", False),
        )

    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent query failed: {str(e)}")


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EvaluationRequest):
    """
    Evaluate a query-answer pair using RAGAS-style metrics.

    Returns:
    - Context Relevance
    - Faithfulness
    - Answer Relevance
    - Overall Score + Grade
    """
    try:
        evaluator = get_evaluator()
        result = await evaluator.evaluate(
            query=request.question,
            answer=request.answer,
            contexts=request.contexts,
        )

        return EvaluationResponse(**result)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ── Session / Memory Routes ───────────────────────────────────


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """List all active conversation sessions."""
    memory = get_memory()
    session_ids = memory.get_session_ids()

    return [
        SessionInfo(**memory.get_session_stats(sid))
        for sid in session_ids
    ]


@router.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    memory = get_memory()
    history = memory.get_history(session_id)

    return ConversationHistoryResponse(
        session_id=session_id,
        messages=history,
    )


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session's history."""
    memory = get_memory()
    memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
