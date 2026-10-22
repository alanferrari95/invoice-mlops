"""Engine B: invoice field extraction, independent of the LLM provider.

PDF -> page filter (pages.py) -> one LLM call on the kept pages' text, constrained to the
InvoiceFields JSON schema -> validated fields + token usage + latency.

The LLM is a pluggable backend (see make_backend):
    EXTRACT_BACKEND=vertex      Gemini on Vertex AI, Google credentials (ADC), no API key
    EXTRACT_BACKEND=gemini-api  Gemini API with GOOGLE_API_KEY
    EXTRACT_BACKEND=ollama      a local model served by Ollama (CPU or your GPU)
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from src.extract.pages import FilteredDocument, filter_pdf

PROMPTS_DIR = Path(__file__).with_name("prompts")
BACKENDS = ("vertex", "gemini-api", "ollama")


def load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.is_file():
        raise ValueError(f"Unknown prompt version {version!r} (no {path.name})")
    return path.read_text(encoding="utf-8").strip()


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thoughts_tokens


class LLMBackend(Protocol):
    model: str

    def generate(self, system: str, text: str) -> tuple[dict, Usage]:
        """Return the InvoiceFields-shaped dict and the token usage of one call."""


@dataclass
class ExtractionResult:
    fields: dict
    usage: Usage
    latency_ms: float
    model: str
    prompt_version: str
    pages_total: int
    pages_kept: list[int]
    doc_sha256: str
    extra: dict = field(default_factory=dict)

    def to_response(self) -> dict:
        return {
            "fields": self.fields,
            "usage": {**asdict(self.usage), "total_tokens": self.usage.total},
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "pages": {"total": self.pages_total, "kept": self.pages_kept},
        }


class Extractor:
    def __init__(self, backend: LLMBackend, prompt_version: str = "v2") -> None:
        self.backend = backend
        self.prompt_version = prompt_version
        self.prompt = load_prompt(prompt_version)

    @property
    def model(self) -> str:
        return self.backend.model

    def extract_filtered(self, doc: FilteredDocument, doc_sha256: str) -> ExtractionResult:
        t0 = time.perf_counter()
        fields, usage = self.backend.generate(self.prompt, doc.text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ExtractionResult(
            fields=fields,
            usage=usage,
            latency_ms=latency_ms,
            model=self.backend.model,
            prompt_version=self.prompt_version,
            pages_total=doc.pages_total,
            pages_kept=doc.pages_kept,
            doc_sha256=doc_sha256,
            extra={"chars_total": doc.chars_total, "chars_kept": doc.chars_kept},
        )

    def extract(self, pdf_bytes: bytes) -> ExtractionResult:
        return self.extract_filtered(filter_pdf(pdf_bytes), hashlib.sha256(pdf_bytes).hexdigest())


def make_backend(name: Optional[str] = None, model: Optional[str] = None) -> LLMBackend:
    name = name or os.getenv("EXTRACT_BACKEND", "vertex")
    if name in ("vertex", "gemini-api"):
        from src.extract.gemini import GeminiBackend, make_genai_client

        return GeminiBackend(make_genai_client(vertex=name == "vertex"),
                             model=model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    if name == "ollama":
        from src.extract.ollama import OllamaBackend

        return OllamaBackend(model=model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    raise ValueError(f"EXTRACT_BACKEND must be one of {BACKENDS}, got {name!r}")
