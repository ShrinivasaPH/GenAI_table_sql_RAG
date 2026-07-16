"""
Table-Aware RAG — FastAPI backend.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000 (frontend is served as static files).
Env:  export OPENAI_API_KEY=sk-...
"""
from dotenv import load_dotenv
load_dotenv()
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.rag import RAGStore

app = FastAPI(title="Table-Aware RAG", version="1.0")
store = RAGStore()

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class ChatRequest(BaseModel):
    question: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "document": store.filename,
        "tables": len(store.tables),
        "chunks": len(store.chunks),
    }


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        return store.ingest_pdf(data, file.filename)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.get("/api/tables/{name}")
def table_preview(name: str, limit: int = 10):
    if name not in store.tables:
        raise HTTPException(404, f"No such table: {name}")
    return store.table_preview(name, limit)


@app.post("/api/chat")
def chat(req: ChatRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "Question is empty.")
    try:
        result = store.answer(q)
    except Exception as e:
        raise HTTPException(500, f"Answer failed: {e}")
    if "error" in result:
        raise HTTPException(409, result["error"])
    return result

@app.delete("/api/document")
def remove_document():
    """Drop the ingested document: SQLite tables, FAISS index, chunks."""
    store.reset()
    return {"status": "cleared"}

# ---- static frontend (mounted last so /api/* wins) ----

@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")
