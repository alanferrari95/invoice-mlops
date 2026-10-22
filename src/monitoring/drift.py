"""Daily drift check: production predictions (BigQuery) vs. training data (GCS).

PSI per feature + KS test on the model score. If any check crosses its
threshold, the training pipeline is re-triggered (with the launcher's guards)
and a JSON alert is published to Pub/Sub.

Runs as the `drift-check` Cloud Run Job, executed daily by Cloud Scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import settings

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]
PREDICTIONS_TABLE = settings.PREDICTIONS_TABLE
EPS = 1e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-uri", default=f"gs://{settings.RAW_BUCKET}/raw/invoices.csv")
    parser.add_argument("--model-uri", default=os.getenv("MODEL_URI", "models/model.joblib"),
                        help="used when --registry-model is not set")
    parser.add_argument("--registry-model", default=os.getenv("REGISTRY_MODEL", ""),
                        help="projects/<p>/locations/<l>/models/<id>; score with the serving model")
    parser.add_argument("--model-alias", default=os.getenv("MODEL_ALIAS", "production"))
    parser.add_argument("--table", default=PREDICTIONS_TABLE)
    parser.add_argument("--window-minutes", type=int, default=24 * 60,
                        help="production window: predictions logged in the last N minutes")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--psi-threshold", type=float, default=0.25)
    parser.add_argument("--ks-pvalue", type=float, default=0.01)
    parser.add_argument("--topic", default=os.getenv("DRIFT_TOPIC", ""),
                        help="projects/<p>/topics/<t>; empty = dry run")
    parser.add_argument("--retrain", action=argparse.BooleanOptionalAction,
                        default=os.getenv("DRIFT_RETRAIN", "0") == "1",
                        help="re-trigger the training pipeline when drift is detected")
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


def resolve_model(args) -> tuple[str, str | None]:
    """(model URI, version label). With the Registry, the version is used to compare
    production scores of *that* model only; scores of older versions are not comparable."""
    if not args.registry_model:
        return args.model_uri, None
    from src import registry

    version = registry.resolve(args.registry_model, args.model_alias)
    if version is None:
        raise RuntimeError(f"No version of {args.registry_model} holds alias '{args.model_alias}'")
    return version.model_uri, version.label


def load_production(table: str, minutes: int, model_version: str | None = None) -> pd.DataFrame:
    from google.cloud import bigquery

    params = [bigquery.ScalarQueryParameter("minutes", "INT64", minutes)]
    version_filter = ""
    if model_version:
        version_filter = "AND model_version = @model_version"
        params.append(bigquery.ScalarQueryParameter("model_version", "STRING", model_version))
    query = f"""
        SELECT {", ".join(FEATURE_NAMES)}, score
        FROM `{table}`
        WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @minutes MINUTE) {version_filter}
    """
    config = bigquery.QueryJobConfig(query_parameters=params)
    return bigquery.Client().query(query, job_config=config).to_dataframe()


def retrain() -> dict:
    from src.launcher import VertexPipelineApi, launch

    result = launch("drift", VertexPipelineApi())
    return {"submitted": result.submitted, "reason": result.reason, "job": result.job}


def publish(topic: str, payload: dict) -> str:
    from google.cloud import pubsub_v1

    data = json.dumps(payload).encode("utf-8")
    return pubsub_v1.PublisherClient().publish(topic, data, kind="drift").result()


def main(argv=None) -> int:
    args = parse_args(argv)
    model_uri, model_version = resolve_model(args)
    prod = load_production(args.table, args.window_minutes, model_version)
    base = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "window_minutes": args.window_minutes,
        "table": args.table,
        "model_version": model_version,
    }

    if len(prod) < args.min_rows:
        report = {**base, "status": "insufficient_data", "n_production": len(prod), "min_rows": args.min_rows}
        print(json.dumps(report))
        return 0

    ref = load_reference(args.reference_uri)
    model = load_model(model_uri)
    ref_scores = model.predict_proba(ref[FEATURE_NAMES])[:, 1]
    prod_scores = prod["score"].dropna().to_numpy()

    report = {**base, **compute_report(ref, prod, ref_scores, prod_scores,
                                       args.psi_threshold, args.ks_pvalue)}

    if report["status"] == "drift" and args.retrain:
        report["retrain"] = retrain()
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
