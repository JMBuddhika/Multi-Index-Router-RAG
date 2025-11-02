from __future__ import annotations
from typing import List, Dict, Any

def simple_chunk(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    max_chunks: int = 5000,   # safety cap
) -> List[str]:
    """Simple sliding-window chunker with a hard cap on chunk count."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        if len(chunks) >= max_chunks:
            break
        end = min(len(text), i + chunk_size)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0:
            i = 0
        if i >= len(text):
            break
    return chunks

def attach_meta(chunks: List[str], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"text": c, "meta": meta} for c in chunks]
