"""Numbers that can be reproduced on a laptop, no cloud: risk model on the gold set and the
page filter on the extraction gold set. These are the only offline numbers in the README.

    python scripts/local_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.gate import gold_metrics  # noqa: E402
from src.extract.pages import filter_pdf  # noqa: E402
from src.train.train import fit  # noqa: E402


def main() -> None:
    model, holdout = fit(pd.read_csv(ROOT / "data" / "invoices.csv"), n_estimators=100, max_depth=4)
    gold = gold_metrics(model, pd.read_csv(ROOT / "data" / "gold" / "risk_gold.csv"))

    gold_dir = ROOT / "data" / "gold" / "extract"
    labels = [json.loads(line) for line in (gold_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
    exact = pages_total = pages_kept = chars_total = chars_kept = 0
    for label in labels:
        doc = filter_pdf((gold_dir / "docs" / f"{label['doc_id']}.pdf").read_bytes())
        exact += doc.pages_kept == [label["invoice_page"]]
        pages_total += doc.pages_total
        pages_kept += len(doc.pages_kept)
        chars_total += doc.chars_total
        chars_kept += doc.chars_kept

    report = {
        "risk_model": {
            "config": {"n_estimators": 100, "max_depth": 4},
            "holdout": {k: round(holdout[k], 4) for k in ("accuracy", "precision", "recall", "f1")} | {"n": holdout["n_test"]},
            "gold": {k: round(v, 4) if isinstance(v, float) else v for k, v in gold.items()},
        },
        "page_filter": {
            "documents": len(labels),
            "exactly_the_invoice_page": exact,
            "pages_sent": f"{pages_kept}/{pages_total}",
            "chars_sent_ratio": round(chars_kept / chars_total, 4),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
