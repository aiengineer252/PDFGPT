"""
Document Routes - PDF upload, listing, and deletion endpoints.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.deps import (
    get_document_registry,
    get_faiss_store,
    get_hybrid_retriever,
    get_ingestion_pipeline,
)
from app.api.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    chunking_strategy: str = Form(default="semantic", description="Chunking strategy: semantic, recursive, context_aware"),
):
    """
    Upload and process a PDF document.

    The document will be:
    1. Saved to disk
    2. Text extracted (including tables)
    3. Chunked using the specified strategy
    4. Embedded and stored in the vector database
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Validate chunking strategy
    valid_strategies = ["semantic", "recursive", "context_aware"]
    if chunking_strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunking strategy. Choose from: {valid_strategies}",
        )

    # Save file
    save_path = settings.UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Saved uploaded file: {save_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Process through ingestion pipeline
    try:
        pipeline = get_ingestion_pipeline()
        result = await pipeline.ingest_pdf(
            file_path=str(save_path),
            chunking_strategy=chunking_strategy,
        )

        # Rebuild BM25 index after adding new documents
        retriever = get_hybrid_retriever()
        retriever.rebuild_bm25()

        # Register document
        registry = get_document_registry()
        registry[result["doc_id"]] = {
            "doc_id": result["doc_id"],
            "file_name": file.filename,
            "upload_time": datetime.now().isoformat(),
            "chunks_count": result["chunks_created"],
            "chunking_strategy": chunking_strategy,
            "file_path": str(save_path),
        }

        return DocumentUploadResponse(**result)

    except Exception as e:
        # Clean up on failure
        if save_path.exists():
            save_path.unlink()
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all processed documents."""
    registry = get_document_registry()

    documents = [
        DocumentInfo(**info) for info in registry.values()
    ]

    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
    )


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document and its vectors from the system."""
    registry = get_document_registry()

    if doc_id not in registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    doc_info = registry[doc_id]

    # Delete vectors from FAISS
    store = get_faiss_store()
    deleted_count = store.delete_by_doc_id(doc_id)
    store.save()

    # Rebuild BM25 index
    retriever = get_hybrid_retriever()
    retriever.rebuild_bm25()

    # Delete file from disk
    file_path = Path(doc_info.get("file_path", ""))
    if file_path.exists():
        file_path.unlink()

    # Remove from registry
    del registry[doc_id]

    return DocumentDeleteResponse(
        doc_id=doc_id,
        vectors_deleted=deleted_count,
        status="deleted",
    )
