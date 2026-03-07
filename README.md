# DeepScholar AI Service

FastAPI + LangGraph multi-agent system for intelligent document analysis, Q&A, and research automation.

## Overview

This service provides:
- **PDF Processing Pipeline**: Extract text, chunk, generate embeddings, store in PGVector
- **Multi-Agent RAG Workflow**: Planner → Clarifier/Researcher/Reader → Writer → Reviewer
- **Vector Search**: Semantic similarity search over article chunks
- **Chat & Q&A**: Answer questions about documents with citations
- **Deep Research**: External search integration (placeholder)

## 🚀 Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 16 with pgvector extension (or use Docker)
- OpenAI API key (optional, currently using mock embeddings)

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Or (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Start service
uvicorn app.main:app --reload
```

Service will be available at `http://localhost:8000/docs` (Swagger UI).

## 📁 Project Structure

```
ai-service/
├── app/
│   ├── main.py                 # FastAPI app setup, router registration
│   ├── api/
│   │   ├── pdf.py              # POST /pdf/upload, GET /pdf/{id}/status
│   │   ├── chatbot.py          # POST /chat/ask, GET /chat/history
│   │   ├── research.py         # POST /research/deep-search
│   │   └── health.py           # GET /health
│   ├── agents/
│   │   ├── planner.py          # Analyze question, decide routing
│   │   ├── clarifier.py        # Clarify ambiguous questions
│   │   ├── researcher.py       # External search (mock)
│   │   ├── reader.py           # Vector DB search
│   │   ├── writer.py           # Synthesize answer with citations
│   │   └── reviewer.py         # Quality check, confidence score
│   ├── workflows/
│   │   ├── states.py           # AgentState Pydantic model
│   │   └── rag_workflow.py     # LangGraph workflow orchestration
│   ├── tools/
│   │   ├── vector_search.py    # Query PGVector, similarity search
│   │   ├── external_search.py  # External research (mock)
│   │   └── citation.py         # Format citations
│   ├── embeddings/
│   │   ├── embedder.py         # Text → embedding vector
│   │   └── vector_store.py     # PGVector operations
│   ├── pdf_pipeline/
│   │   ├── loader.py           # Handle file uploads
│   │   ├── extractor.py        # PDF → text (PyPDF2)
│   │   ├── chunker.py          # Text → chunks with overlap
│   │   └── metadata.py         # Extract title, abstract, authors
│   ├── schemas/
│   │   ├── request.py          # ChatRequest, ResearchRequest
│   │   └── response.py         # ChatResponse
│   └── core/
│       ├── config.py           # Settings (env vars, API keys)
│       ├── logger.py           # Logging setup
│       └── memory.py           # In-process memory stores
├── data/
│   └── uploads/                # Temporary PDF uploads
├── requirements.txt
├── .env.example
├── Dockerfile
└── sample.pdf                  # Example for local testing
```

## 🔧 Configuration

### Environment Variables (.env)

```env
DATABASE_URL=postgresql://deepscholar:deepscholar@localhost:5432/deepscholar
OPENAI_API_KEY=                 # Leave empty to use mock embeddings
CORS_ALLOW_ORIGINS=["*"]       # Restrict in production
EMBEDDING_DIMENSION=8           # Size of embedding vectors (mock)
MAX_CHAT_HISTORY=20            # Max messages per session
UPLOAD_DIR=./data/uploads      # PDF storage directory
```

## 📚 API Endpoints

### Health
- `GET /health` - Service status and database connectivity

### PDF Processing
- `POST /api/pdf/upload` - Upload and process PDF
  - FormData: `article_id`, `file` (PDF)
  - Returns: `task_id`, metadata, chunk count
- `GET /api/pdf/{task_id}/status` - Check processing status

### Chatbot & Q&A
- `POST /api/chat/ask` - Submit question about article
  - Body: `question`, `article_id`, `session_id` (optional)
  - Returns: answer, citations, confidence_score, review feedback
- `GET /api/chat/history/{session_id}` - Retrieve chat history

### Research
- `POST /api/research/deep-search` - External research query (mock)
  - Body: `query`
  - Returns: search results with sources

## 🤖 Multi-Agent Workflow

### Flow Diagram

```
User Question → Planner (analyze, route)
                   ↓
        ┌──────────┴──────────┬─────────────┐
        ↓                     ↓              ↓
    Clarifier?          Researcher      Reader
    (disambiguate)    (external search) (vector DB)
        ↓                     ↓              ↓
        └──────────┬──────────┴─────────────┘
                   ↓
                Writer (synthesize + format)
                   ↓
                Reviewer (quality check)
                   ↓
    ┌──────────────┴──────────────┐
    ↓                             ↓
  Rewrite?                      Accept
(iterate if < 0.7 confidence)    ↓
    ↓                           END
  Writer
```

### Agent Responsibilities

**PlannerAgent**
- Parse question for ambiguity (< 4 words or vague phrases → need_clarification)
- Detect if external search needed (keywords: latest, benchmark, compare, etc)
- Identify focus sections (Method, Results, Conclusion, etc)

**ClarifierAgent**
- If needed_clarification=True, ask user to refine question
- Return clarification prompt; exit workflow

**ReaderAgent**
- Query PGVector for relevant chunks from article
- Filter by focus_sections if specified
- Return top-5 similar chunks with scores

**ResearcherAgent**
- Call external_search tool if needed_external_search=True
- Return background info, comparisons, related work

**WriterAgent**
- Synthesize vector context + external context
- Format answer with evidence and citations
- Keep response concise (<250 words)
- Append revision notes if reviewer requested rewrite

**ReviewerAgent**
- Check: has context? has citations? is concise?
- Confidence score: 0.0-1.0 based on above factors
- If score < 0.7 and iterations < max, request rewrite
- Otherwise, approve final answer

### State Schema

```python
class AgentState(BaseModel):
    # Input
    question: str
    article_id: int
    session_id: Optional[str] = None
    
    # Planner decisions
    need_clarification: bool = False
    need_external_search: bool = False
    focus_sections: List[str] = []
    
    # Retrieved data
    vector_context: List[Dict] = []        # From Reader
    external_context: List[Dict] = []      # From Researcher
    citations: List[Dict] = []             # Built by Writer
    
    # Outputs
    draft_answer: Optional[str] = None
    reviewed_answer: Optional[str] = None
    clarification_question: Optional[str] = None
    
    # Control
    review_feedback: Optional[str] = None
    confidence_score: float = 0.0
    iteration_count: int = 0
    max_iterations: int = 2
    
    memory_id: Optional[str] = None
```

## 📦 Key Technologies

- **Framework**: FastAPI 0.115+, Uvicorn
- **Workflow**: LangGraph 0.2+, LangChain 0.3+
- **PDF**: PyPDF2 3.0+ (text extraction only, no OCR)
- **Database**: psycopg2 (PostgreSQL), pgvector
- **Embeddings**: Mock hash-based vectors (8-dimensional)
- **Config**: Pydantic Settings 2.4+

## 🔄 PDF Processing Pipeline

1. **Upload** - Accept PDF file via multipart form
2. **Extract** - PyPDF2 extracts text from all pages
3. **Infer Metadata** - Heuristic extraction of title, authors, abstract
4. **Chunk** - Split text into 900-char chunks with 150-char overlap
5. **Embed** - Generate 8-d mock vector for each chunk
6. **Store** - Insert into PostgreSQL + pgvector table
7. **Index** - Enable similarity search via `<=>` operator

## 🧪 Development Commands

```bash
# Start with hot reload
uvicorn app.main:app --reload

# Start with specific host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Check import validity
python -c "from app.main import app; print(app.title)"

# Test PDF upload (Linux/macOS)
curl -F "article_id=1" -F "file=@sample.pdf" \
  http://localhost:8000/api/pdf/upload

# View API docs
# http://localhost:8000/docs
```

## 🐳 Docker Usage

```bash
# Build image
docker build -t deepscholar-ai .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  deepscholar-ai
```

Or via docker-compose:
```bash
docker compose up ai-service
```

## 📊 Vector Database

### PGVector Setup

Table structure created automatically on first request:

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER UNIQUE REFERENCES article_chunks(id),
    embedding vector(8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Similarity Search

Uses cosine distance (`<=>` operator):
```sql
SELECT * FROM embeddings
WHERE embedding <=> query_vector < distance_threshold
ORDER BY embedding <=> query_vector
LIMIT 5;
```

## ⚙️ Performance Notes

- **Mock Embeddings**: Hash-based for fast local testing (not for production)
- **In-Memory History**: Persists only during session; restarts lose history
- **PDF Chunking**: 900 chars per chunk helps balance token efficiency and context
- **Vector Dimension**: Set to 8 for scaffolding; use 1536+ for real embeddings

## 📤 Production Roadmap

- [ ] Replace mock embeddings with OpenAI/HuggingFace
- [ ] Implement real external search (Semantic Scholar, Google Scholar)
- [ ] Add persistent memory backend (database or Redis)
- [ ] Integrate real LLM for agents (Claude, GPT-4)
- [ ] Add OCR for scanned PDFs
- [ ] Implement async job queue (Celery)
- [ ] Add caching layer (Redis)
- [ ] Authentication & rate limiting
- [ ] Structured logging
- [ ] Monitoring & alerting

## 🔗 Integration Points

- **Backend**: Expects PGVector table in shared PostgreSQL
- **Frontend**: Open for CORS from http://localhost:3000
- **OpenAI**: API key for real embeddings (optional)

## 📝 License

Internal use - Capstone Project 2026
