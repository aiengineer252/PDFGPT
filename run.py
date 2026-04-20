"""
PDFGPT - Advanced RAG System Entry Point.

Run the server with:
    python run.py

Or with uvicorn directly:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
