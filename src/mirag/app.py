from __future__ import annotations
import os
from typing import List, Dict, Any
from fastapi import FastAPI, Body
from dotenv import load_dotenv

from .llm_groq import GroqLLM
from .embeddings import Embeddings
from .retrievers.text_pdf_code import TextPdfCodeIndex
from .retrievers.sql_duckdb import DuckSQL
from .stores import rrf_fusion
from .router import MultiIndexRouter

load_dotenv()

DATA_DOCS = os.getenv("DATA_DOCS", "data/docs")
DATA_PDFS = os.getenv("DATA_PDFS", "data/pdfs")
DATA_CODE = os.getenv("DATA_CODE", "data/code")
DATA_TABLES = os.getenv("DATA_TABLES", "data/tables")

app = FastAPI(title="Multi-Index Router RAG")

_emb = Embeddings()
_textpdfcode = TextPdfCodeIndex(_emb)
_textpdfcode.build(DATA_DOCS, DATA_PDFS, DATA_CODE)

_sql = DuckSQL(":memory:")
_sql.ingest_csv_glob(DATA_TABLES)

_router = MultiIndexRouter()
_llm = GroqLLM()

def _fmt_citation(p: Dict[str, Any]) -> str:
    meta = p.get("meta", {})
    src = meta.get("source")
    if src == "pdf":
        return f"{os.path.basename(meta.get('file',''))} p.{meta.get('page')}"
    elif src == "doc":
        return os.path.basename(meta.get('file',''))
    elif src == "code":
        return f"{os.path.basename(meta.get('file',''))}:{meta.get('symbol')}"
    else:
        return str(meta)

def synthesize_answer(question: str, contexts: List[Dict[str, Any]], sql_block: Dict[str, Any] | None) -> str:
    ctx_text = ""
    for i, p in enumerate(contexts, start=1):
        ctx_text += f"[{i}] ({_fmt_citation(p)})\n{p['text']}\n\n"

    sql_txt = ""
    if sql_block:
        if "error" in sql_block and sql_block["error"]:
            sql_txt = f"SQL:\n{sql_block['sql']}\nERROR: {sql_block['error']}\n"
        else:
            sql_txt = f"SQL:\n{sql_block['sql']}\nRESULTS (first 10):\n"
            rows = sql_block.get("rows", [])[:10]
            cols = sql_block.get("columns", [])
            sql_txt += " | ".join(cols) + "\n"
            for r in rows:
                sql_txt += " | ".join(map(str, r)) + "\n"

    system = (
        "You are a precise assistant that writes concise answers grounded in the provided context.\n"
        "Rules:\n"
        "- Cite evidence inline using bracket numbers like [1], [2] that refer to the snippets.\n"
        "- If SQL results are provided, incorporate them and keep numbers exact.\n"
        "- If insufficient information, say so and suggest what is missing.\n"
        "- Keep the answer factual and avoid speculation."
    )
    user = f"Question:\n{question}\n\nContext snippets:\n{ctx_text}\n{sql_txt}\n\nWrite the final answer with citations."
    return _llm.chat(system, user, temperature=0.1)

def retrieve_docs(question: str, topk: int = 6) -> List[Dict[str, Any]]:
    r = _textpdfcode.search(question, topk=topk)
    return [p for _score, p in r]

@app.post("/ask")
def ask(payload: Dict[str, Any] = Body(...)):
    q = payload.get("question", "").strip()
    topk = int(payload.get("topk", 6))
    if not q:
        return {"error": "missing question"}

    decision = _router.decide(q)
    ctxs: List[Dict[str,Any]] = []
    sql_block = None

    if decision.route == "sql":
        sql_block = _sql.query(q)
        ctxs = retrieve_docs(q, topk=3)

    elif decision.route in ("doc", "pdf", "code"):
        ctxs = retrieve_docs(q, topk=topk)

    elif decision.route == "hybrid":
        order = decision.hybrid_order or ["sql","doc"]
        for op in order:
            if op == "sql":
                sql_block = _sql.query(q)
            else:
                if not ctxs:
                    ctxs = retrieve_docs(q, topk=topk)

    answer = synthesize_answer(q, ctxs, sql_block)
    return {
        "route": decision.route,
        "reason": decision.reason,
        "hybrid_order": decision.hybrid_order,
        "sql": sql_block,
        "citations": [ _fmt_citation(p) for p in ctxs ],
        "answer": answer
    }

@app.get("/health")
def health():
    return {"ok": True}
