"""Gemini backend (google-genai SDK) for the extractor: structured output via response_schema.

Two ways to authenticate, chosen by EXTRACT_BACKEND:
- vertex: Vertex AI with Application Default Credentials (a service account on Cloud Run /
  Vertex, `gcloud auth application-default login` on a laptop). No API key exists anywhere.
- gemini-api: the Gemini Developer API with GOOGLE_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from src.extract.extractor import Usage
from src.extract.schema import InvoiceFields

DEFAULT_MODEL = "gemini-3.8-flash"
# No current GA Gemini Flash model is served from the europe-west4 regional endpoint
# (checked 2026-10-22); the `eu` multi-region keeps processing inside the EU.
DEFAULT_LOCATION = "eu"


def make_genai_client(vertex: bool = True, project: Optional[str] = None, location: Optional[str] = None):
    from google import genai

    if not vertex:
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    from src import settings

    return genai.Client(
        vertexai=True,
        project=project or settings.PROJECT,
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


class GeminiBackend:
    def __init__(self, client: Any, model: str = DEFAULT_MODEL, thinking_level: Optional[str] = None) -> None:
        self.client = client
        self.model = model
        self.thinking_level = thinking_level or os.getenv("GEMINI_THINKING_LEVEL", "LOW")

    def _config(self, system: str):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=InvoiceFields,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
        )

    def generate(self, system: str, text: str) -> tuple[dict, Usage]:
        response = self.client.models.generate_content(model=self.model, contents=text, config=self._config(system))
        return _parse(response), _usage(response)
