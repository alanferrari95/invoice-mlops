"""Train a binary risk classifier on synthetic invoices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.cloud import storage
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-uri", type=str, default="data/invoices.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--model-dir", type=str, default="models")
    return parser.parse_args()


def _split_gs(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://") :]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def download_from_gcs(gcs_uri: str, local_path: Path) -> Path:
    bucket_name, blob_name = _split_gs(gcs_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    print(f"Downloaded {gcs_uri} -> {local_path}")
    return local_path


def upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        return
    bucket_name, blob_name = _split_gs(gcs_uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    print(f"Uploaded {local_path} -> {gcs_uri}")


def resolve_data_path(data_uri: str) -> Path:
    if data_uri.startswith("gs://"):
        return download_from_gcs(data_uri, Path("/tmp/invoices.csv"))
    return Path(data_uri)


def _build_classifier(n_estimators: int, max_depth: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )


def fit(df: pd.DataFrame, n_estimators: int, max_depth: int):
    """Train on an 80/20 stratified split; return the model and its holdout metrics."""
    X = df[FEATURE_NAMES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = _build_classifier(n_estimators, max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_names": FEATURE_NAMES,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    return model, metrics


def save(model, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(resolve_data_path(args.data_uri))
    model, metrics = fit(df, args.n_estimators, args.max_depth)
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"{key}: {metrics[key]:.4f}")

    if args.model_dir.startswith("gs://"):
        out_dir = Path("/tmp/model-out")
        save(model, metrics, out_dir)
        prefix = args.model_dir.rstrip("/")
        upload_to_gcs(out_dir / "model.joblib", f"{prefix}/model.joblib")
        upload_to_gcs(out_dir / "metrics.json", f"{prefix}/metrics.json")
    else:
        save(model, metrics, Path(args.model_dir))


if __name__ == "__main__":
    main()
