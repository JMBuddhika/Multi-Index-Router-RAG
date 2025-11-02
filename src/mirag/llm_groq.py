from __future__ import annotations
import os
from typing import Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class GroqLLM:
    def __init__(self, model: str | None = None):
        if not GROQ_API_KEY:
            raise RuntimeError("Missing GROQ_API_KEY in environment")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model or GROQ_MODEL

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        json_sys = system + "\nAlways respond with STRICT JSON only. No prose."
        content = self.chat(json_sys, user, temperature=temperature)
        import json, re
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.S)
            if not m:
                raise ValueError(f"Model did not return JSON: {content[:300]}")
            return json.loads(m.group(0))
