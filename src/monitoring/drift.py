"""Daily drift check: production predictions (BigQuery) vs. training data (GCS).

PSI per feature + KS test on the model score. If any check crosses its
threshold, a JSON alert is published to Pub/Sub.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]
PREDICTIONS_TABLE = "invoice-mlops.invoice_mlops.predictions"
EPS = 1e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-uri", default="gs://invoice-mlops-raw/raw/invoices.csv")
    parser.add_argument("--model-uri", default="models/model.joblib")
    parser.add_argument("--table", default=PREDICTIONS_TABLE)
    parser.add_argument("--hours", type=int, default=24, help="production window size")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--psi-threshold", type=float, default=0.25)
    parser.add_argument("--ks-pvalue", type=float, default=0.01)
    parser.add_argument("--topic", default="", help="projects/<p>/topics/<t>; empty = dry run")
    return parser.parse_args(argv)


def psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with bins cut on reference quantiles."""
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)
    # inner cut points; unique() collapses repeated edges of discrete features (hour, is_new_vendor)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(edges) + 1
    ref_pct = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=n_bins) / len(reference)
    prod_pct = np.bincount(np.searchsorted(edges, production, side="right"), minlength=n_bins) / len(production)
    ref_pct = np.clip(ref_pct, EPS, None)
    prod_pct = np.clip(prod_pct, EPS, None)
    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def compute_report(ref: pd.DataFrame, prod: pd.DataFrame, ref_scores, prod_scores,
                   psi_threshold: float, ks_pvalue: float) -> dict:
    features = {}
    for name in FEATURE_NAMES:
        value = psi(ref[name], prod[name])
        features[name] = {"psi": round(value, 4), "drift": value > psi_threshold}
    ks = ks_2samp(ref_scores, prod_scores)
    score = {
        "ks_stat": round(float(ks.statistic), 4),
        "p_value": float(ks.pvalue),
        "drift": bool(ks.pvalue < ks_pvalue),
    }
    drifted = [n for n, f in features.items() if f["drift"]] + (["score"] if score["drift"] else [])
    return {
        "status": "drift" if drifted else "ok",
        "drifted": drifted,
        "n_reference": len(ref),
        "n_production": len(prod),
        "features": features,
        "score": score,
    }


def _download(uri: str, dest: Path) -> Path:
    from google.cloud import storage

    bucket, _, blob = uri[len("gs://") :].partition("/")
    storage.Client().bucket(bucket).blob(blob).download_to_filename(str(dest))
    return dest


def load_reference(uri: str) -> pd.DataFrame:
    path = _download(uri, Path("/tmp/reference.csv")) if uri.startswith("gs://") else Path(uri)
    return pd.read_csv(path)


def load_model(uri: str):
    path = _download(uri, Path("/tmp/drift_model.joblib")) if uri.startswith("gs://") else Path(uri)
    return joblib.load(path)


def load_production(table: str, hours: int) -> pd.DataFrame:
    from google.cloud import bigquery

    query = f"""
        SELECT {", ".join(FEATURE_NAMES)}, score
        FROM `{table}`
        WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("hours", "INT64", hours)]
    )
    return bigquery.Client().query(query, job_config=config).to_dataframe()


def publish(topic: str, payload: dict) -> str:
    from google.cloud import pubsub_v1

    data = json.dumps(payload).encode("utf-8")
    return pubsub_v1.PublisherClient().publish(topic, data, kind="drift").result()


def main(argv=None) -> int:
    args = parse_args(argv)
    prod = load_production(args.table, args.hours)
    base = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.hours,
        "table": args.table,
    }

    if len(prod) < args.min_rows:
        report = {**base, "status": "insufficient_data", "n_production": len(prod), "min_rows": args.min_rows}
        print(json.dumps(report))
        return 0

    ref = load_reference(args.reference_uri)
    model = load_model(args.model_uri)
    ref_scores = model.predict_proba(ref[FEATURE_NAMES])[:, 1]
    prod_scores = prod["score"].dropna().to_numpy()

    report = {**base, **compute_report(ref, prod, ref_scores, prod_scores,
                                       args.psi_threshold, args.ks_pvalue)}
    print(json.dumps(report))

    if report["status"] == "drift":
        if args.topic:
            msg_id = publish(args.topic, report)
            print(f"Published drift alert {msg_id} to {args.topic}", file=sys.stderr)
        else:
            print("Drift detected (dry run, no topic set)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
