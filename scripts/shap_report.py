"""SHAP explanations for the risk model: three plots for docs/.

Trains the model exactly as the pipeline does (same data, same hyperparameters, same split)
and explains it on the gold set with TreeSHAP.

    pip install -r requirements-docs.txt
    python scripts/shap_report.py            # -> docs/img/shap_*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.gate import FEATURE_NAMES  # noqa: E402
from src.train.train import fit  # noqa: E402

OUT = ROOT / "docs" / "img"


def main() -> None:
    model, _ = fit(pd.read_csv(ROOT / "data" / "invoices.csv"), n_estimators=100, max_depth=4)
    gold = pd.read_csv(ROOT / "data" / "gold" / "risk_gold.csv")
    X = gold[FEATURE_NAMES]
    explanation = shap.TreeExplainer(model)(X)
    OUT.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(explanation, show=False)
    plt.title("SHAP values per feature (gold set, 120 invoices)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=150)
    plt.close()

    shap.plots.bar(explanation, show=False)
    plt.title("Mean |SHAP value| (global importance)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar.png", dpi=150)
    plt.close()

    proba = model.predict_proba(X)[:, 1]
    idx = int(proba.argmax())
    shap.plots.waterfall(explanation[idx], show=False)
    plt.title(f"Why gold invoice {int(gold.iloc[idx]['invoice_id'])} scores {proba[idx]:.2f}")
    plt.tight_layout()
    plt.savefig(OUT / "shap_waterfall_high_risk.png", dpi=150)
    plt.close()

    importance = pd.Series(abs(explanation.values).mean(axis=0), index=FEATURE_NAMES).sort_values(ascending=False)
    print("mean |SHAP| (log-odds):")
    print(importance.round(4).to_string())


if __name__ == "__main__":
    main()
