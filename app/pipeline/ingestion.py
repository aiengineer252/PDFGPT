"""
Ingestion Pipeline - Full PDF → Chunks → Embeddings → Vector Store pipeline.

Handles:
1. PDF text extraction (with page-level metadata)
2. Smart chunking (configurable strategy)
3. Embedding generation
4. Vector store persistence
5. BM25 index building
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

from app.config import settings
from app.core.chunking.semantic_chunker import Chunk, SemanticChunker
from app.core.chunking.recursive_chunker import RecursiveChunker
from app.core.chunking.context_aware import ContextAwareChunker
from app.core.embeddings.embedding_manager import EmbeddingManager
from app.core.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    PDF → Extract Text → Chunk → Embed → Store
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        embedding_manager: EmbeddingManager,
        chunking_strategy: str = settings.CHUNKING_STRATEGY,
    ):
        self.faiss_store = faiss_store
        self.embedding_manager = embedding_manager
        self.chunking_strategy = chunking_strategy

        # Initialize chunkers
        self._semantic_chunker = None
        self._recursive_chunker = RecursiveChunker()
        self._context_aware_chunker = ContextAwareChunker()

    def _get_semantic_chunker(self) -> SemanticChunker:
        """Lazy-load semantic chunker (shares embedding model)."""
        if self._semantic_chunker is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._semantic_chunker = SemanticChunker(embedding_model=model)
        return self._semantic_chunker

    async def ingest_pdf(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a PDF file into the RAG system.

        Args:
            file_path: Path to the PDF file.
            doc_id: Optional document ID. Auto-generated if not provided.
            chunking_strategy: Override chunking strategy for this document.

        Returns:
            Ingestion result dict with stats.
        """
        doc_id = doc_id or str(uuid.uuid4())[:8]
        strategy = chunking_strategy or self.chunking_strategy
        file_path = Path(file_path)

        logger.info(f"Starting ingestion: {file_path.name} (doc_id={doc_id}, strategy={strategy})")

        # Step 1: Extract text from PDF
        pages = self._extract_pdf_text(str(file_path))
        if not pages:
            raise ValueError(f"Could not extract text from {file_path.name}")

        total_text_length = sum(len(p["text"]) for p in pages)
        logger.info(f"Extracted {len(pages)} pages ({total_text_length} chars)")

        # Step 2: Chunk the text
        chunks = self._chunk_text(pages, strategy, {
            "doc_id": doc_id,
            "source_file": file_path.name,
            "total_pages": len(pages),
        })

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        logger.info(f"Created {len(chunks)} chunks using '{strategy}' strategy")

        # Step 3: Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = self.embedding_manager.embed_documents(
            chunk_texts, show_progress=True
        )

        # Step 4: Store in FAISS
        metadata_list = [
            {**c.metadata, "chunk_id": c.chunk_id, "page_numbers": c.page_numbers}
            for c in chunks
        ]

        n_added = self.faiss_store.add(
            vectors=embeddings,
            texts=chunk_texts,
            metadata_list=metadata_list,
            doc_id=doc_id,
        )

        # Step 5: Save to disk
        self.faiss_store.save()

        result = {
            "doc_id": doc_id,
            "file_name": file_path.name,
            "total_pages": len(pages),
            "total_characters": total_text_length,
            "chunks_created": len(chunks),
            "vectors_stored": n_added,
            "chunking_strategy": strategy,
            "embedding_dimension": self.embedding_manager.embedding_dimension,
            "status": "success",
        }

        logger.info(f"Ingestion complete: {result}")
        return result

    def _extract_pdf_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page of the PDF."""
        pages = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""

                    # Also try to extract tables
                    tables = page.extract_tables() or []
                    table_text = ""
                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " | ".join(
                                    str(cell) if cell else "" for cell in row
                                )
                                table_text += row_text + "\n"

                    full_text = text
                    if table_text:
                        full_text += "\n\n[TABLE]\n" + table_text

                    if full_text.strip():
                        pages.append({
                            "text": full_text.strip(),
                            "page_number": i + 1,
                            "has_tables": len(tables) > 0,
                        })

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

        return pages

    def _chunk_text(
        self,
        pages: List[Dict[str, Any]],
        strategy: str,
        base_metadata: dict,
    ) -> List[Chunk]:
        """Chunk extracted text using the specified strategy."""

        if strategy == "context_aware":
            return self._context_aware_chunker.chunk_pages(pages, base_metadata)

        # For semantic and recursive, concatenate all text first
        full_text = "\n\n".join(
            f"[Page {p['page_number']}]\n{p['text']}" for p in pages
        )

        if strategy == "semantic":
            chunker = self._get_semantic_chunker()
            chunks = chunker.chunk(full_text, base_metadata)
        elif strategy == "recursive":
            chunks = self._recursive_chunker.chunk(full_text, base_metadata)
        else:
            # Default to recursive
            chunks = self._recursive_chunker.chunk(full_text, base_metadata)

        # Attempt to assign page numbers to chunks
        for chunk in chunks:
            chunk.page_numbers = self._infer_page_numbers(chunk.content, pages)

        return chunks

    def _infer_page_numbers(
        self, chunk_text: str, pages: List[Dict[str, Any]]
    ) -> List[int]:
        """Try to figure out which pages a chunk came from."""
        page_numbers = []

        # Check for page markers
        import re
        markers = re.findall(r'\[Page (\d+)\]', chunk_text)
        if markers:
            page_numbers = list(set(int(m) for m in markers))
        else:
            # Check overlap with each page
            chunk_words = set(chunk_text.lower().split()[:20])
            for page in pages:
                page_words = set(page["text"].lower().split()[:50])
                overlap = len(chunk_words & page_words)
                if overlap > 5:
                    page_numbers.append(page["page_number"])

        return sorted(page_numbers)[:3]  # Max 3 page refs
