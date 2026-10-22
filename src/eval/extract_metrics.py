"""Per-field precision and recall for invoice extraction, and the prompt promotion gate.

For each document and field, after normalisation:
- TP: a value was predicted and it equals the gold value;
- FP: a value was predicted and it is wrong (including a value where gold is null);
- FN: gold has a value and the prediction is missing or wrong.
precision = TP / (TP + FP)   "when it answers, is it right?"
recall    = TP / (TP + FN)   "how much of what is there does it get?"
A wrong value counts against both: it is a false answer and a missed one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from src.extract.schema import FIELDS

AMOUNT_TOLERANCE = 0.01


def _norm_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^0-9a-z\u00c0-\u024f]+", "", text)


def matches(field: str, predicted, gold) -> bool:
    if predicted is None or gold is None:
        return False
    if field == "total_amount":
        try:
            return abs(float(predicted) - float(gold)) <= AMOUNT_TOLERANCE
        except (TypeError, ValueError):
            return False
    if field in ("invoice_date", "currency"):
        return str(predicted).strip().upper() == str(gold).strip().upper()
    return _norm_text(predicted) == _norm_text(gold)


def _empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def field_report(predictions: dict[str, dict], gold: dict[str, dict]) -> dict:
    """predictions / gold: doc_id -> {field: value}. Docs missing from predictions count as all-null."""
    counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    for doc_id, truth in gold.items():
        pred = predictions.get(doc_id, {})
        for f in FIELDS:
            p = None if _empty(pred.get(f)) else pred.get(f)
            g = None if _empty(truth.get(f)) else truth.get(f)
            ok = matches(f, p, g)
            if p is not None:
                counts[f]["tp" if ok else "fp"] += 1
            if g is not None and not ok:
                counts[f]["fn"] += 1

    def prf(c):
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**c, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    per_field = {f: prf(c) for f, c in counts.items()}
    micro = prf({k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")})
    return {"fields": per_field, "micro": micro, "n_docs": len(gold)}


@dataclass
class PromptDecision:
    promote: bool
    reason: str


def decide_prompt(micro: dict, champion_micro_f1: Optional[float], min_precision: float,
                  min_recall: float, max_regression: float) -> PromptDecision:
    p, r, f1 = micro["precision"], micro["recall"], micro["f1"]
    if p < min_precision:
        return PromptDecision(False, f"micro precision {p:.4f} < {min_precision}")
    if r < min_recall:
        return PromptDecision(False, f"micro recall {r:.4f} < {min_recall}")
    if champion_micro_f1 is not None and f1 < champion_micro_f1 - max_regression:
        return PromptDecision(False, f"micro f1 {f1:.4f} regresses vs champion {champion_micro_f1:.4f}")
    vs = "no champion" if champion_micro_f1 is None else f"champion micro f1 {champion_micro_f1:.4f}"
    return PromptDecision(True, f"micro P={p:.4f} R={r:.4f} F1={f1:.4f}; {vs}")
