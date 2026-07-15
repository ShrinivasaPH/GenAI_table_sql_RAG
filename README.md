# Table-Aware RAG (FastAPI + HTML/JS)

Industry-style rebuild of the Streamlit POC. Same pipeline, real web architecture:
a Python API backend and a plain HTML/CSS/JS frontend talking over REST — the
pattern used in production RAG products (the frontend could later be swapped
for React without touching the backend).

## Pipeline

1. **Ingest PDF** → pdfplumber extracts prose and tables separately
2. **Tables** → cleaned DataFrames → SQLite (one SQL table per extracted table)
3. **Prose** → chunks → OpenAI embeddings → FAISS index
4. **Query** → LLM router decides: `sql` / `vector` / `hybrid`
5. `sql` path: LLM writes a SQLite SELECT against the real schema → execute → answer.
   `vector` path: top-k chunks → answer with context. `hybrid` uses both.
   If generated SQL fails, it gracefully falls back to the vector path.

## Structure

```
table-rag/
├── backend/
│   ├── main.py     # FastAPI: HTTP endpoints + serves the frontend
│   └── rag.py      # RAG engine: ingestion, routing, retrieval (no HTTP)
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js      # fetch-based upload, chat, evidence rendering
├── requirements.txt
└── README.md
```

Separation of concerns: `rag.py` knows nothing about HTTP, `main.py` knows
nothing about retrieval, and the frontend only ever talks JSON. This is the
answer to "why not Streamlit" in an interview: independent layers, standard
REST contract, deployable behind nginx/gunicorn, testable in isolation.

## API

| Method | Path                 | Purpose                                        |
|--------|----------------------|------------------------------------------------|
| GET    | `/api/health`        | Backend + API-key status, ingested doc counts  |
| POST   | `/api/ingest`        | multipart PDF upload → extraction summary      |
| GET    | `/api/tables/{name}` | Preview rows of an extracted SQL table         |
| POST   | `/api/chat`          | `{question}` → `{answer, route, sql, rows, chunks}` |

Every chat response carries its evidence: the route chosen, the exact SQL
executed with result rows, and/or retrieved chunks with L2 distances. The UI
renders this in a collapsible panel — useful for demos and debugging.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # Windows: set OPENAI_API_KEY=sk-...
uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000 — the frontend is served by FastAPI itself,
so there are no CORS issues and only one process to run.

## POC limitations (be upfront in interviews)

- Single in-memory document store (`RAGStore`), no auth, no multi-tenancy —
  production would add sessions/user scoping and a persistent vector DB
  (pgvector, Qdrant, Pinecone).
- LLM-generated SQL is guarded (SELECT-only, single statement, in-memory DB
  with only extracted tables) but a production system would add a read-only
  connection and query timeouts.
- Character-based chunking; semantic/recursive chunking is the upgrade path.
- Table extraction uses a first-row-header heuristic; complex merged-cell
  tables need a layout model (e.g. table transformers) in production.
