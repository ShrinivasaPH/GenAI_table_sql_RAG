"""
Table-Aware RAG engine (framework-agnostic).

Pipeline (same as the Streamlit POC, now decoupled from any UI):
  1. Ingest PDF  -> pdfplumber extracts prose + tables separately
  2. Tables      -> cleaned DataFrames -> SQLite (one SQL table each)
  3. Prose       -> chunks -> OpenAI embeddings -> FAISS index
  4. Query       -> LLM router decides: sql / vector / hybrid
  5. sql path    -> LLM writes SQLite against the real schema -> execute
     vector path -> top-k chunks -> answer with context

The FastAPI layer (main.py) owns HTTP; this module owns state + logic.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import threading

import numpy as np
import pandas as pd
import faiss
import pdfplumber
from openai import OpenAI

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 900          # chars
CHUNK_OVERLAP = 150
TOP_K = 4

_client: OpenAI | None = None


def client() -> OpenAI:
    """Lazy init so the server can boot without OPENAI_API_KEY set."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------- ingestion

def sanitize(name: str, fallback: str) -> str:
    name = re.sub(r"\W+", "_", str(name).strip().lower()).strip("_")
    return name or fallback


def table_to_df(raw: list) -> pd.DataFrame | None:
    """First row = header heuristic; drop empty rows/cols; coerce numerics."""
    if not raw or len(raw) < 2:
        return None
    header = [sanitize(h, f"col_{i}") for i, h in enumerate(raw[0])]
    df = pd.DataFrame(raw[1:], columns=header)
    df = df.replace("", np.nan).dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return None
    for col in df.columns:
        cleaned = df[col].astype(str).str.replace(r"[,\s₹$%]", "", regex=True)
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().mean() > 0.7:          # mostly numeric column
            df[col] = num
    return df


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def embed(texts: list[str]) -> np.ndarray:
    res = client().embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in res.data], dtype="float32")


def chat(system: str, user: str) -> str:
    res = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()


# ---------------------------------------------------------------- store

class RAGStore:
    """Holds everything for one ingested document. In-memory, single-tenant POC."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.chunks: list[str] = []
        self.tables: list[str] = []
        self.index: faiss.Index | None = None
        self.filename: str | None = None

    # -------- ingestion

    def ingest_pdf(self, data: bytes, filename: str) -> dict:
        with self.lock:
            self.reset()
            self.filename = filename
            prose_parts: list[str] = []

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for pno, page in enumerate(pdf.pages, 1):
                    for tno, raw in enumerate(page.extract_tables(), 1):
                        df = table_to_df(raw)
                        if df is None:
                            continue
                        tname = f"p{pno}_t{tno}"
                        df.to_sql(tname, self.conn, index=False)
                        self.tables.append(tname)
                    if txt := page.extract_text():
                        prose_parts.append(txt)

            self.chunks = chunk_text("\n".join(prose_parts))
            if self.chunks:
                vecs = embed(self.chunks)
                self.index = faiss.IndexFlatL2(vecs.shape[1])
                self.index.add(vecs)

            return {
                "filename": filename,
                "tables": [self.table_preview(t) for t in self.tables],
                "chunk_count": len(self.chunks),
            }

    # -------- schema / previews

    def schema(self) -> str:
        lines = []
        for t in self.tables:
            cols = pd.read_sql(f'SELECT * FROM "{t}" LIMIT 3', self.conn)
            lines.append(f'Table "{t}" columns: {list(cols.columns)}')
            lines.append(f"Sample rows: {cols.to_dict(orient='records')}")
        return "\n".join(lines)

    def table_preview(self, name: str, limit: int = 10) -> dict:
        df = pd.read_sql(f'SELECT * FROM "{name}" LIMIT {int(limit)}', self.conn)
        total = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        return {
            "name": name,
            "columns": list(df.columns),
            "rows": json.loads(df.to_json(orient="values")),
            "total_rows": total,
        }

    # -------- routing + answering

    def route(self, question: str) -> str:
        if self.tables and not self.chunks:
            return "sql"
        if self.chunks and not self.tables:
            return "vector"
        decision = chat(
            "You are a query router. Decide how to answer a question about a document. "
            "Respond with exactly one word: 'sql' if it needs numbers/aggregates/lookups "
            "from tables, 'vector' if it needs prose/descriptive context, "
            "'hybrid' if it needs both.",
            f"Available SQL tables:\n{self.schema()}\n\nQuestion: {question}",
        ).lower()
        return decision if decision in ("sql", "vector", "hybrid") else "hybrid"

    def sql_context(self, question: str) -> tuple[str, str, list]:
        """Returns (context, executed_sql, rows)."""
        sql = chat(
            "Write a single SQLite SELECT statement to answer the question. "
            "Use only the tables/columns in the schema. Return ONLY the SQL, "
            "no markdown, no explanation.",
            f"Schema:\n{self.schema()}\n\nQuestion: {question}",
        )
        sql = re.sub(r"^```(sql)?|```$", "", sql.strip(), flags=re.M).strip()
        if not re.match(r"(?is)^\s*select\b", sql) or ";" in sql.rstrip(";"):
            raise ValueError(f"Router produced non-SELECT SQL: {sql!r}")
        df = pd.read_sql(sql, self.conn)
        rows = json.loads(df.to_json(orient="records"))
        ctx = f"SQL executed:\n{sql}\n\nResult:\n{df.to_string(index=False)}"
        return ctx, sql, rows

    def vector_context(self, question: str) -> tuple[str, list[dict]]:
        """Returns (context, [{text, score}])."""
        if not self.index:
            return "", []
        qv = embed([question])
        dist, idx = self.index.search(qv, min(TOP_K, len(self.chunks)))
        hits = [{"text": self.chunks[i], "score": round(float(d), 4)}
                for d, i in zip(dist[0], idx[0]) if i != -1]
        ctx = "\n\n---\n\n".join(h["text"] for h in hits)
        return ctx, hits

    def answer(self, question: str) -> dict:
        if not self.tables and not self.chunks:
            return {"error": "No document ingested yet. Upload a PDF first."}

        route = self.route(question)
        evidence: dict = {"route": route}
        ctx_parts = []

        if route in ("sql", "hybrid"):
            try:
                ctx, sql, rows = self.sql_context(question)
                ctx_parts.append(ctx)
                evidence["sql"] = sql
                evidence["rows"] = rows
            except Exception as e:
                evidence["sql_error"] = str(e)
                if route == "sql":
                    route = "vector"          # graceful fallback
                    evidence["route"] = "vector (sql failed)"

        if route in ("vector", "hybrid") or "sql_error" in evidence:
            ctx, hits = self.vector_context(question)
            if ctx:
                ctx_parts.append(ctx)
                evidence["chunks"] = hits

        full_ctx = "\n\n=====\n\n".join(ctx_parts)
        ans = chat(
            "Answer using ONLY the provided context. Quote exact numbers from SQL "
            "results verbatim — never estimate. If the context is insufficient, say so.",
            f"Context:\n{full_ctx}\n\nQuestion: {question}",
        )
        return {"answer": ans, **evidence}
