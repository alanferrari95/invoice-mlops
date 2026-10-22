"""Fake Gemini client: 'reads' the prompt text and answers from the gold labels.

It returns the gold fields of whichever gold document's vendor name appears in the text it
receives, so it only answers correctly if the page filter kept the invoice page.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.extract.schema import FIELDS, InvoiceFields

GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold" / "extract"


def gold_labels() -> list[dict]:
    return [json.loads(l) for l in (GOLD_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l]


class FakeModels:
    def __init__(self, labels, corrupt=None):
        self.labels = labels
        self.corrupt = corrupt or {}
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        match = next((l for l in self.labels if l["vendor_name"] in contents), None)
        fields = {f: (match or {}).get(f) for f in FIELDS}
        fields.update(self.corrupt)
        usage = SimpleNamespace(prompt_token_count=len(contents) // 4, candidates_token_count=40,
                                thoughts_token_count=10)
        return SimpleNamespace(parsed=InvoiceFields(**fields), text=json.dumps(fields), usage_metadata=usage)


class FakeGeminiClient:
    def __init__(self, labels=None, corrupt=None):
        self.models = FakeModels(labels if labels is not None else gold_labels(), corrupt)
