"""Engine B: invoice field extraction with Gemini structured output.

PDF -> page filter (pages.py) -> one Gemini call on the kept pages' text with
response_schema=InvoiceFields -> validated fields + token usage + latency.

The Gemini client is injected, so tests use a fake and never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.extract.pages import FilteredDocument, filter_pdf
from src.extract.schema import InvoiceFields

DEFAULT_MODEL = "gemini-3.8-flash"
# No current GA Gemini Flash model is served from the europe-west4 regional endpoint;
# the `eu` multi-region keeps processing inside the EU.
DEFAULT_LOCATION = "eu"
PROMPTS_DIR = Path(__file__).with_name("prompts")


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


def make_client(project: Optional[str] = None, location: Optional[str] = None):
    from google import genai

    return genai.Client(
        vertexai=True,
        project=project or os.getenv("GOOGLE_CLOUD_PROJECT", "invoice-mlops"),
        location=location or os.getenv("GEMINI_LOCATION", DEFAULT_LOCATION),
    )


def _usage(response: Any) -> Usage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return Usage()
    return Usage(
        prompt_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
        thoughts_tokens=int(getattr(meta, "thoughts_token_count", 0) or 0),
    )


def _parse(response: Any) -> dict:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, InvoiceFields):
        return parsed.model_dump()
    return InvoiceFields.model_validate(json.loads(response.text)).model_dump()


class GeminiExtractor:
    def __init__(self, client: Any, model: str = DEFAULT_MODEL, prompt_version: str = "v2",
                 thinking_level: Optional[str] = None) -> None:
        self.client = client
        self.model = model
        self.prompt_version = prompt_version
        self.prompt = load_prompt(prompt_version)
        self.thinking_level = thinking_level or os.getenv("GEMINI_THINKING_LEVEL", "LOW")

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=self.prompt,
            response_mime_type="application/json",
            response_schema=InvoiceFields,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
        )

    def extract_filtered(self, doc: FilteredDocument, doc_sha256: str) -> ExtractionResult:
        t0 = time.perf_counter()
        response = self.client.models.generate_content(model=self.model, contents=doc.text, config=self._config())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ExtractionResult(
            fields=_parse(response),
            usage=_usage(response),
            latency_ms=latency_ms,
            model=self.model,
            prompt_version=self.prompt_version,
            pages_total=doc.pages_total,
            pages_kept=doc.pages_kept,
            doc_sha256=doc_sha256,
            extra={"chars_total": doc.chars_total, "chars_kept": doc.chars_kept},
        )

    def extract(self, pdf_bytes: bytes) -> ExtractionResult:
        return self.extract_filtered(filter_pdf(pdf_bytes), hashlib.sha256(pdf_bytes).hexdigest())
