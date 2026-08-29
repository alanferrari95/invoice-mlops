"""Train a binary risk classifier on synthetic invoices."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]
TARGET = "risk"
RANDOM_STATE = 42

MODEL_CARD = """# Tarjeta de modelo — riesgo de revisión

Este modelo predice **riesgo de revisión manual** como etiqueta binaria (0/1):
si una factura debería pasar por un analista antes de pagarse.

**No predice fraude real.** Un 1 no significa que haya delito; solo que hay
señales de inconsistencia o rareza que justifican revisión humana.

**Features:** `amount`, `lines_sum`, `hour`, `is_new_vendor`,
`vendor_avg_amount`, `amount_vs_avg_ratio`. No se usan identificadores
(`invoice_id`, `vendor_id`, `vendor_name`).

La etiqueta `risk` se **bootstrappea con reglas** (ratio vs. promedio del
proveedor, descuadre amount/lines_sum, proveedor nuevo, hora nocturna) y
luego se **invierte en un 8%** de filas (ruido de etiqueta).

**Límites conocidos:** los datos son sintéticos; el rendimiento no se
generaliza a facturas reales. Si `vendor_avg_amount` se calcula mal en
producción (p. ej. incluyendo la factura actual o usando otro universo de
proveedores) hay **riesgo de leakage / drift** y las predicciones dejan de
ser comparables al entrenamiento.
"""


def _build_classifier():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=RANDOM_STATE)


def main() -> None:
    df = pd.read_csv(Path("data") / "invoices.csv")
    X = df[FEATURE_NAMES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = _build_classifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1: {f1:.4f}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "model.joblib")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": FEATURE_NAMES,
    }
    (models_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "model_card.md").write_text(MODEL_CARD.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
