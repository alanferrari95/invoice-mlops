"""Ollama backend: a local model (CPU or GPU) with no cloud account and no key.

Uses Ollama's /api/chat with `format` set to the InvoiceFields JSON schema, so the model is
constrained to the same structured output as Gemini.

    ollama pull qwen2.5:7b
    EXTRACT_BACKEND=ollama OLLAMA_MODEL=qwen2.5:7b uvicorn src.api.main:app
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from src.extract.extractor import Usage
from src.extract.schema import InvoiceFields

Post = Callable[[str, dict, float], dict]


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    import requests

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:7b", host: Optional[str] = None,
                 timeout: float = 300.0, post: Post = _http_post) -> None:
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.post = post

    def generate(self, system: str, text: str) -> tuple[dict, Usage]:
        body = self.post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                "format": InvoiceFields.model_json_schema(),
                "stream": False,
                "options": {"temperature": 0},
            },
            self.timeout,
        )
        fields = InvoiceFields.model_validate(json.loads(body["message"]["content"])).model_dump()
        usage = Usage(prompt_tokens=int(body.get("prompt_eval_count", 0)), output_tokens=int(body.get("eval_count", 0)))
        return fields, usage
