# 🚀 PDFGPT - Advanced RAG PDF Query System

A **production-grade Advanced Retrieval-Augmented Generation (RAG)** system for querying PDF documents. Built with 100% free resources — no paid APIs required.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDFGPT System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  PDF Upload   │───►│  Smart       │───►│  Embedding       │  │
│  │  (pdfplumber) │    │  Chunking    │    │  (BGE Model)     │  │
│  └──────────────┘    └──────────────┘    └───────┬──────────┘  │
│                                                   │              │
│                                          ┌────────▼─────────┐  │
│                                          │   FAISS Vector    │  │
│                                          │   Store + BM25    │  │
│                                          └────────┬─────────┘  │
│                                                   │              │
│  ┌──────────────┐    ┌──────────────┐    ┌────────▼─────────┐  │
│  │  User Query   │───►│  Query       │───►│  Hybrid          │  │
│  │              │    │  Rewriter    │    │  Retrieval       │  │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘  │
│                                                   │              │
│                      ┌──────────────┐    ┌────────▼─────────┐  │
│                      │  Context     │◄───│  Cross-Encoder   │  │
│                      │  Compression │    │  Re-ranking      │  │
│                      └──────┬───────┘    └──────────────────┘  │
│                             │                                   │
│  ┌──────────────┐   ┌──────▼───────┐    ┌──────────────────┐  │
│  │  Guardrails   │◄──│  LLM Answer  │    │  RAGAS           │  │
│  │  Validation   │   │  Generation  │───►│  Evaluation      │  │
│  └──────────────┘   └──────────────┘    └──────────────────┘  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Conversation │    │  Agentic RAG │    │  Streaming       │  │
│  │  Memory       │    │  Agent       │    │  Responses       │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⭐ Features (13 Advanced RAG Components)

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Smart Chunking** | Semantic, Recursive, and Context-Aware splitting |
| 2 | **BGE Embeddings** | BAAI/bge-small-en-v1.5 (free, local, ~130MB) |
| 3 | **FAISS Vector Store** | High-performance local similarity search with persistence |
| 4 | **Hybrid Retrieval** | Vector + BM25 keyword search with Reciprocal Rank Fusion |
| 5 | **Query Rewriting** | LLM-powered expansion, reformulation, and HyDE |
| 6 | **Multi-Vector Retrieval** | Raw + summary embeddings per chunk |
| 7 | **Cross-Encoder Re-ranking** | ms-marco-MiniLM-L-6-v2 for precision re-scoring |
| 8 | **Context Compression** | LLM-based extraction + redundancy removal |
| 9 | **RAGAS Evaluation** | Context relevance, faithfulness, answer relevance scoring |
| 10 | **Conversation Memory** | Session-based history with context window management |
| 11 | **Guardrails** | Groundedness checking, hallucination detection, confidence scoring |
| 12 | **Streaming** | Server-Sent Events (SSE) for real-time token streaming |
| 13 | **Agentic RAG** | AI agent dynamically selects retrieval strategy per query |

---

## 📁 Project Structure

```
PDFGPT/
├── app/
│   ├── main.py                          # FastAPI application
│   ├── config.py                        # Pydantic settings
│   ├── api/
│   │   ├── schemas.py                   # Request/response models
│   │   ├── deps.py                      # Dependency injection
│   │   └── routes/
│   │       ├── documents.py             # PDF upload/delete/list
│   │       ├── query.py                 # Query/stream/agent/evaluate
│   │       └── health.py               # Health check
│   ├── core/
│   │   ├── chunking/
│   │   │   ├── semantic_chunker.py      # Embedding-based boundary detection
│   │   │   ├── recursive_chunker.py     # Hierarchical splitting with overlap
│   │   │   └── context_aware.py         # Structure-preserving chunking
│   │   ├── embeddings/
│   │   │   └── embedding_manager.py     # BGE embedding singleton
│   │   ├── vectorstore/
│   │   │   └── faiss_store.py           # FAISS with metadata & persistence
│   │   ├── retrieval/
│   │   │   ├── hybrid_retriever.py      # Vector + BM25 + RRF
│   │   │   ├── query_rewriter.py        # Query expansion + HyDE
│   │   │   ├── multi_vector.py          # Multi-representation retrieval
│   │   │   └── reranker.py              # Cross-encoder re-ranking
│   │   ├── compression/
│   │   │   └── context_compressor.py    # LLM-based compression
│   │   ├── memory/
│   │   │   └── conversation_memory.py   # Session history management
│   │   ├── guardrails/
│   │   │   └── safety.py                # Output validation
│   │   ├── evaluation/
│   │   │   └── retrieval_eval.py        # RAGAS-style metrics
│   │   ├── agent/
│   │   │   └── rag_agent.py             # Agentic RAG orchestration
│   │   └── llm/
│   │       └── llm_manager.py           # Groq/Ollama LLM wrapper
│   └── pipeline/
│       ├── ingestion.py                 # PDF → chunks → vectors
│       └── query_pipeline.py            # query → retrieve → answer
├── data/
│   ├── uploads/                         # Uploaded PDFs
│   └── vectorstores/                    # Persisted FAISS indexes
├── .env.example                         # Environment template
├── requirements.txt                     # Dependencies
├── run.py                               # Entry point
└── README.md                           # You are here
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Free Groq API Key** from [console.groq.com](https://console.groq.com)

### 1. Clone & Install

```bash
cd "d:\ML Projects\PDFGPT"

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env and add your Groq API key
# GROQ_API_KEY=gsk_your_key_here
```

### 3. Run the Server

```bash
python run.py
```

Server starts at: **http://localhost:8000**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. First-Time Model Download

On first startup, the system downloads:
- **BGE Embedding Model** (~130MB) — runs locally
- **Cross-Encoder Reranker** (~80MB) — runs locally

This only happens once. Subsequent startups are fast.

---

## 📡 API Reference

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload & process a PDF |
| `GET` | `/api/documents` | List all documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document |

### Querying

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Standard RAG query (full pipeline) |
| `POST` | `/api/query/stream` | Streaming response (SSE) |
| `POST` | `/api/query/agent` | Agentic RAG (AI decides strategy) |
| `POST` | `/api/query/evaluate` | Evaluate a query-answer pair |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/query/sessions` | List conversation sessions |
| `GET` | `/api/query/sessions/{id}/history` | Get session history |
| `DELETE` | `/api/query/sessions/{id}` | Clear session |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |

---

## 🧪 Usage Examples

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@my_document.pdf" \
  -F "chunking_strategy=semantic"
```

**Response:**
```json
{
  "doc_id": "a1b2c3d4",
  "file_name": "my_document.pdf",
  "total_pages": 24,
  "chunks_created": 87,
  "vectors_stored": 87,
  "chunking_strategy": "semantic",
  "status": "success"
}
```

### Query with Full Pipeline

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "session_id": "user-123",
    "enable_evaluation": true
  }'
```

**Response:**
```json
{
  "answer": "Based on the document, the key findings are...",
  "sources": [
    {
      "text": "The study found that...",
      "metadata": {"page_number": 5, "section": "Results"},
      "score": 0.92,
      "retrieval_method": "vector+bm25"
    }
  ],
  "validation": {
    "confidence_score": 0.87,
    "is_valid": true
  },
  "evaluation": {
    "overall_score": 0.85,
    "grade": "A"
  }
}
```

### Agentic Query

```bash
curl -X POST http://localhost:8000/api/query/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the entire document",
    "session_id": "user-123"
  }'
```

### Streaming Response (JavaScript)

```javascript
const response = await fetch('http://localhost:8000/api/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "What is the main conclusion?",
    session_id: "user-123"
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.token) {
        process.stdout.write(data.token);
      }
      if (data.done) break;
    }
  }
}
```

---

## 🎨 Frontend Integration Guide

### React / Next.js Integration

#### 1. Install Dependencies

```bash
npm install axios
# or for streaming:
npm install eventsource-parser
```

#### 2. API Service (TypeScript)

```typescript
// src/services/pdfgptApi.ts

const API_BASE = 'http://localhost:8000/api';

export interface QueryRequest {
  question: string;
  session_id?: string;
  doc_id?: string;
  enable_evaluation?: boolean;
}

export interface QueryResponse {
  answer: string;
  sources: Array<{
    text: string;
    metadata: Record<string, any>;
    score: number;
  }>;
  validation: {
    confidence_score: number;
    is_valid: boolean;
  };
  evaluation?: {
    overall_score: number;
    grade: string;
  };
}

// Upload PDF
export async function uploadPDF(file: File, strategy = 'semantic') {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('chunking_strategy', strategy);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

// Standard Query
export async function queryDocuments(request: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return res.json();
}

// Streaming Query
export async function* streamQuery(
  request: QueryRequest
): AsyncGenerator<string> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.token) yield data.token;
        if (data.done) return;
      }
    }
  }
}

// List Documents
export async function listDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  return res.json();
}

// Delete Document
export async function deleteDocument(docId: string) {
  const res = await fetch(`${API_BASE}/documents/${docId}`, {
    method: 'DELETE',
  });
  return res.json();
}
```

#### 3. React Chat Component Example

```tsx
// src/components/ChatInterface.tsx

import { useState, useRef } from 'react';
import { queryDocuments, streamQuery } from '../services/pdfgptApi';

export function ChatInterface() {
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const sessionId = useRef(`session-${Date.now()}`);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);

    // Streaming version
    let fullResponse = '';
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    for await (const token of streamQuery({
      question: input,
      session_id: sessionId.current,
    })) {
      fullResponse += token;
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
        return updated;
      });
    }

    setIsStreaming(false);
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          placeholder="Ask about your documents..."
          disabled={isStreaming}
        />
        <button onClick={handleSubmit} disabled={isStreaming}>
          {isStreaming ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

#### 4. Frontend Setup Steps

1. Create your React/Next.js project:
   ```bash
   npx create-next-app@latest pdfgpt-frontend
   cd pdfgpt-frontend
   ```

2. Copy the API service file from above

3. Build your UI components using the API service

4. Configure the API base URL in your `.env`:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000/api
   ```

5. Start the frontend:
   ```bash
   npm run dev
   ```

6. The backend CORS is already configured to allow `localhost:3000` and `localhost:5173`.

---

## ⚙️ Configuration Reference

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider: `groq` or `ollama` |
| `GROQ_API_KEY` | — | Free API key from console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `CHUNK_SIZE` | `512` | Target chunk size (characters) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CHUNKING_STRATEGY` | `semantic` | Default: `semantic`, `recursive`, `context_aware` |
| `TOP_K_RETRIEVAL` | `20` | Candidates to retrieve |
| `TOP_K_RERANK` | `5` | Top results after re-ranking |
| `HYBRID_ALPHA` | `0.5` | Vector vs keyword weight (0=keyword, 1=vector) |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `["localhost:3000","localhost:5173"]` | Allowed frontend origins |

---

## 🔧 Using Ollama (Fully Local, No API Key)

To run everything locally without any API key:

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Update `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

---

## 📊 How the Pipeline Works

### Ingestion Pipeline

```
PDF ──► pdfplumber (text + tables) ──► Smart Chunking ──► BGE Embeddings ──► FAISS Store
```

### Query Pipeline (8 stages)

```
1. Context Resolution   │ Resolve pronouns using conversation history
2. Query Rewriting      │ LLM improves vague queries
3. Hybrid Retrieval     │ FAISS (semantic) + BM25 (keyword) + RRF merge
4. Re-ranking           │ Cross-encoder scores each query-doc pair
5. Compression          │ LLM extracts only relevant sentences
6. Answer Generation    │ LLM generates answer from compressed context
7. Guardrails           │ Groundedness check + hallucination detection
8. Evaluation           │ RAGAS metrics: relevance, faithfulness, quality
```

### Agentic Pipeline

```
Query ──► Agent analyzes ──► Decides action:
                              ├── Retrieve (standard pipeline)
                              ├── Rewrite + Retrieve (vague queries)
                              ├── Answer Directly (greetings, general)
                              ├── Summarize (broad requests)
                              └── Clarify (too ambiguous)
```

---

## 🆓 All Free Resources Used

| Component | Resource | Cost |
|-----------|----------|------|
| LLM | Groq API (llama-3.3-70b) | **Free** (rate limited) |
| Embeddings | BAAI/bge-small-en-v1.5 | **Free** (local) |
| Reranker | cross-encoder/ms-marco-MiniLM | **Free** (local) |
| Vector DB | FAISS | **Free** (local) |
| BM25 | rank-bm25 | **Free** (local) |
| PDF Parse | pdfplumber | **Free** (local) |

---

## 📝 License

MIT License - feel free to use for any purpose.
