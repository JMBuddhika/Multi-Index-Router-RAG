from __future__ import annotations
from typing import Literal, List
from pydantic import BaseModel
from .llm_groq import GroqLLM

Route = Literal["doc", "pdf", "code", "sql", "hybrid"]

class RouteDecision(BaseModel):
    route: Route
    hybrid_order: List[Literal["sql","doc","pdf","code"]] = []
    reason: str

class MultiIndexRouter:
    def __init__(self):
        self.llm = GroqLLM()

    def decide(self, question: str) -> RouteDecision:
        system = (
            "You are a routing classifier for a multi-index RAG system.\n"
            "Choices:\n"
            " - 'sql'  : if question asks for metrics, numbers, counts, aggregates, tables, dates, filters.\n"
            " - 'code' : if question is about functions, classes, errors, or implementation details in code.\n"
            " - 'pdf'  : if question likely refers to a PDF document (formal reports, whitepapers, manuals).\n"
            " - 'doc'  : for general knowledge in text/markdown/html docs.\n"
            " - 'hybrid': if it clearly needs BOTH structured data (sql) and unstructured docs (doc/pdf/code).\n"
            "Return JSON with keys: route, hybrid_order (list), reason."
        )
        user = f"Question: {question}\nRespond JSON now."
        j = self.llm.chat_json(system, user, temperature=0.0)
        return RouteDecision.model_validate(j)
