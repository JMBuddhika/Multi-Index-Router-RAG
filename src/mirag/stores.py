from __future__ import annotations
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np

class DenseStore:
    """
    Simple FAISS cosine-sim index with L2 vectors normalized.
    """
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine if normalized
        self.vectors = None
        self.payloads: List[Dict[str, Any]] = []

    def add(self, vecs: List[List[float]], payloads: List[Dict[str, Any]]):
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        if self.vectors is None:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        self.index.add(arr)
        self.payloads.extend(payloads)

    def search(self, vec: List[float], topk: int = 8) -> List[Tuple[float, Dict[str, Any]]]:
        q = np.array([vec], dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(q, topk)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            results.append((float(score), self.payloads[idx]))
        return results

def rrf_fusion(run_lists: List[List[Tuple[float, Dict[str, Any]]]], k: int = 8, c: float = 60.0):
    from collections import defaultdict
    agg = defaultdict(float)
    seen = {}
    for run in run_lists:
        for rank, (_score, payload) in enumerate(run, start=1):
            key = (payload.get("meta", {}).get("source"), payload.get("meta", {}).get("id"))
            agg[key] += 1.0 / (c + rank)
            seen[key] = payload
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(score, seen[key]) for key, score in ranked]
