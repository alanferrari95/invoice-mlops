"""Promotion gate for the risk model: score a candidate on the gold set and
decide whether it may replace the model currently serving (the champion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]
TARGET = "risk"


@dataclass
class Decision:
    promote: bool
    reason: str


def gold_metrics(model, gold: pd.DataFrame) -> dict:
    y_true = gold[TARGET]
    y_pred = model.predict(gold[FEATURE_NAMES])
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_gold": int(len(gold)),
    }


def decide(
    candidate_f1: float,
    champion_f1: Optional[float],
    min_f1: float,
    max_regression: float,
) -> Decision:
    """Promote only if the candidate clears the absolute bar and does not lose more
    than max_regression F1 against the champion on the same gold set."""
    if candidate_f1 < min_f1:
        return Decision(False, f"gold f1={candidate_f1:.4f} below min_f1={min_f1}")
    if champion_f1 is not None and candidate_f1 < champion_f1 - max_regression:
        return Decision(
            False,
            f"gold f1={candidate_f1:.4f} regresses vs champion {champion_f1:.4f} "
            f"(max_regression={max_regression})",
        )
    vs = "no comparable champion" if champion_f1 is None else f"champion f1={champion_f1:.4f}"
    return Decision(True, f"gold f1={candidate_f1:.4f} >= min_f1={min_f1}; {vs}")
