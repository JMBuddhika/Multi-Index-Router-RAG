from __future__ import annotations
import os, re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm import tqdm

from ..embeddings import Embeddings
from ..chunkers import simple_chunk, attach_meta
from ..stores import DenseStore

TEXT_EXT = {".txt", ".md", ".html", ".htm"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".cs", ".php", ".rb"}

# Safety limits (can be tuned via env)
MAX_FILE_MB       = int(os.getenv("MAX_FILE_MB", "20"))         # skip files above this size on disk
MAX_DOC_CHARS     = int(os.getenv("MAX_DOC_CHARS", "2000000"))  # hard truncate large plain/html docs
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "5000"))
EMB_BATCH_SIZE    = int(os.getenv("EMB_BATCH_SIZE", "256"))

class TextPdfCodeIndex:
    def __init__(self, emb: Embeddings, dim: int = 384):
        self.emb = emb
        self.store = DenseStore(dim=dim)
        self._built = False

    def _too_big(self, p: Path) -> bool:
        try:
            return p.stat().st_size > MAX_FILE_MB * 1024 * 1024
        except Exception:
            return False

    def _read_textlike(self, p: Path) -> str:
        if p.suffix.lower() in {".html", ".htm"}:
            html = p.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            for s in soup(["script", "style"]):
                s.extract()
            text = soup.get_text(separator="\n")
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS]
        return text

    def _read_pdf(self, p: Path) -> List[Dict[str, Any]]:
        reader = PdfReader(str(p))
        all_chunks: List[Dict[str, Any]] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            chunks = simple_chunk(text, 900, 120, max_chunks=MAX_CHUNKS_PER_FILE)
            all_chunks += attach_meta(chunks, {"source": "pdf", "file": str(p), "page": i+1, "id": f"{p.name}#p{i+1}"})
        return all_chunks

    def _read_code(self, p: Path) -> List[Dict[str, Any]]:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        # naive "block" splitter across many languages
        blocks = re.split(r"\n(?=[a-zA-Z_].{0,120}\{|\s*def\s|\s*class\s)", raw)
        chunks: List[Dict[str, Any]] = []
        for i, b in enumerate(blocks):
            snippet = b.strip()
            if not snippet:
                continue
            meta = {"source": "code", "file": str(p), "symbol": f"block_{i}", "id": f"{p.name}#b{i}"}
            for c in simple_chunk(snippet, 800, 100, max_chunks=MAX_CHUNKS_PER_FILE):
                chunks.append({"text": c, "meta": meta})
        return chunks

    def _add_payloads_in_batches(self, payloads: List[Dict[str, Any]]):
        """Encode and add to the FAISS store in small batches to keep memory low."""
        if not payloads:
            return
        texts = [x["text"] for x in payloads]
        for i in range(0, len(texts), EMB_BATCH_SIZE):
            batch_payloads = payloads[i:i+EMB_BATCH_SIZE]
            batch_texts = texts[i:i+EMB_BATCH_SIZE]
            vecs = self.emb.encode(batch_texts)
            self.store.add(vecs, batch_payloads)

    def build(self, docs_dir: str, pdfs_dir: str, code_dir: str):
        # docs (txt/md/html)
        for p in tqdm(list(Path(docs_dir).glob("**/*")), desc="Indexing docs"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_EXT:
                continue
            if self._too_big(p):
                continue
            txt = self._read_textlike(p)
            payloads = attach_meta(
                simple_chunk(txt, 900, 120, max_chunks=MAX_CHUNKS_PER_FILE),
                {"source": "doc", "file": str(p), "id": p.name}
            )
            self._add_payloads_in_batches(payloads)

        # pdfs
        for p in tqdm(list(Path(pdfs_dir).glob("**/*.pdf")), desc="Indexing pdfs"):
            if not p.is_file():
                continue
            if self._too_big(p):
                continue
            payloads = self._read_pdf(p)
            self._add_payloads_in_batches(payloads)

        # code
        for p in tqdm(list(Path(code_dir).glob("**/*")), desc="Indexing code"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in CODE_EXT:
                continue
            if self._too_big(p):
                continue
            payloads = self._read_code(p)
            self._add_payloads_in_batches(payloads)

        self._built = True

    def search(self, query: str, topk: int = 8) -> List[Tuple[float, Dict[str, Any]]]:
        vec = self.emb.encode([query])[0]
        return self.store.search(vec, topk=topk)
