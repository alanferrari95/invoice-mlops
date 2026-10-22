"""Page filtering before the LLM call.

Only pages that look like the invoice itself are sent to Gemini; cover letters, terms and
remittance slips are dropped. Input tokens are the main cost driver, so this is the cheapest
optimisation in the module. If no page scores above the threshold, every page is kept
(never send nothing).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

# (pattern, weight). Titles and totals weigh most; generic money words weigh little because
# cover letters and terms pages mention amounts too.
SIGNALS = [
    (r"\b(invoice|factura|tax invoice)\b", 3.0),
    (r"\b(invoice no|n[ºo°]\.? de factura|reference)\b", 2.0),
    (r"\b(total|amount due|total factura)\b", 2.0),
    (r"\b(vat|iva|tax|base imponible|subtotal|net)\b", 1.5),
    (r"\b(bill to|cliente|customer)\b", 1.0),
    (r"\d+[.,]\d{2}\b", 0.2),
]
NEGATIVE = [
    (r"\b(terms and conditions|jurisdiction|remittance advice|dear customer|detach)\b", -4.0),
]
THRESHOLD = 5.0


@dataclass
class FilteredDocument:
    text: str
    pages_total: int
    pages_kept: list[int]  # 1-based page numbers sent to the LLM
    chars_total: int

    @property
    def chars_kept(self) -> int:
        return len(self.text)


def page_texts(pdf_bytes: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf_bytes)).pages]


def score_page(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for pattern, weight in SIGNALS:
        score += weight * min(len(re.findall(pattern, lowered)), 5)
    for pattern, weight in NEGATIVE:
        score += weight * len(re.findall(pattern, lowered))
    return score


def filter_pages(texts: list[str], threshold: float = THRESHOLD) -> FilteredDocument:
    kept = [i for i, t in enumerate(texts, start=1) if score_page(t) >= threshold]
    if not kept:
        kept = list(range(1, len(texts) + 1))
    body = "\n\n".join(f"--- page {i} ---\n{texts[i - 1]}" for i in kept)
    return FilteredDocument(body, len(texts), kept, sum(len(t) for t in texts))


def filter_pdf(pdf_bytes: bytes, threshold: float = THRESHOLD) -> FilteredDocument:
    return filter_pages(page_texts(pdf_bytes), threshold)
