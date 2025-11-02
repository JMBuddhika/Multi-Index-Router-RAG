from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer

_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embeddings:
    def __init__(self, model_name: str = _EMB_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=False)
        return [e.tolist() for e in embs]
