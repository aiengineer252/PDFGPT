"""
API Schemas - Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Document Schemas ──────────────────────────────────────────


class DocumentUploadResponse(BaseModel):
    """Response after uploading and processing a PDF."""
    doc_id: str
    file_name: str
    total_pages: int
    total_characters: int
    chunks_created: int
    vectors_stored: int
    chunking_strategy: str
    embedding_dimension: int
    status: str


class DocumentInfo(BaseModel):
    """Information about a processed document."""
    doc_id: str
    file_name: str
    upload_time: str
    chunks_count: int
    chunking_strategy: str


class DocumentListResponse(BaseModel):
    """Response listing all documents."""
    documents: List[DocumentInfo]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    doc_id: str
    vectors_deleted: int
    status: str


# ── Query Schemas ─────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    session_id: str = Field(default="default", description="Session ID for conversation memory")
    doc_id: Optional[str] = Field(default=None, description="Filter by specific document ID")
    enable_rewrite: bool = Field(default=True, description="Enable query rewriting")
    enable_rerank: bool = Field(default=True, description="Enable re-ranking")
    enable_compression: bool = Field(default=True, description="Enable context compression")
    enable_guardrails: bool = Field(default=True, description="Enable output validation")
    enable_evaluation: bool = Field(default=False, description="Enable RAGAS-style evaluation")


class SourceInfo(BaseModel):
    """Information about a source chunk."""
    text: str
    metadata: Dict[str, Any] = {}
    score: float = 0.0
    retrieval_method: str = "unknown"


class QueryResponse(BaseModel):
    """Response from the RAG query pipeline."""
    answer: str
    sources: List[SourceInfo] = []
    search_query: Optional[str] = None
    validation: Dict[str, Any] = {}
    evaluation: Dict[str, Any] = {}
    pipeline_metadata: Dict[str, Any] = {}


# ── Agent Query Schemas ────────────────────────────────────────


class AgentQueryRequest(BaseModel):
    """Request for agentic RAG query."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")
    doc_id: Optional[str] = None


class AgentQueryResponse(BaseModel):
    """Response from the agentic RAG pipeline."""
    answer: str
    sources: List[SourceInfo] = []
    action_taken: str = ""
    reasoning: List[str] = []
    needs_clarification: bool = False


# ── Evaluation Schemas ────────────────────────────────────────


class EvaluationRequest(BaseModel):
    """Request to evaluate a query-answer pair."""
    question: str
    answer: str
    contexts: List[str] = []


class EvaluationResponse(BaseModel):
    """Evaluation results."""
    overall_score: float
    context_relevance: Dict[str, Any] = {}
    faithfulness: Dict[str, Any] = {}
    answer_relevance: Dict[str, Any] = {}
    grade: str = ""
    num_contexts: int = 0


# ── Session Schemas ────────────────────────────────────────────


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    message_count: int
    user_messages: int
    assistant_messages: int
    has_summary: bool


class ConversationHistoryResponse(BaseModel):
    """Conversation history."""
    session_id: str
    messages: List[Dict[str, str]]


# ── Health Check ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    """System health status."""
    status: str = "healthy"
    version: str = "1.0.0"
    llm_provider: str
    embedding_model: str
    total_vectors: int = 0
    total_documents: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
