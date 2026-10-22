"""Week 4: hyperparameter runs as Vertex AI Training custom jobs, tracked in Vertex AI Experiments.

Each config runs the training container (Dockerfile.train) as a CPU custom job that writes
model.joblib + metrics.json to GCS. When the job succeeds, its params, holdout metrics and
gold-set metrics are logged as one run of the experiment, so the runs can be compared side by
side in the console (Vertex AI > Experiments > risk-hparams).

    python scripts/run_experiments.py                  # the 3 default configs
    python scripts/run_experiments.py --configs 100:4 300:2
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import settings  # noqa: E402
from src.eval.gate import gold_metrics  # noqa: E402

PROJECT = settings.PROJECT
LOCATION = settings.REGION
IMAGE = f"{settings.IMAGE_REPO}/train:latest"
SA = settings.WORK_SA
STAGING = f"gs://{settings.ARTIFACTS_BUCKET}/staging"
DEFAULT_CONFIGS = ["100:4", "200:3", "50:6"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="n_estimators:max_depth")
    parser.add_argument("--experiment", default="risk-hparams")
    parser.add_argument("--data-uri", default=f"gs://{settings.RAW_BUCKET}/raw/invoices.csv")
    parser.add_argument("--gold", default="data/gold/risk_gold.csv")
    parser.add_argument("--machine-type", default="e2-standard-4")
    parser.add_argument("--image", default=IMAGE)
    return parser.parse_args()


def read_gcs(uri: str, dest: Path) -> Path:
    from google.cloud import storage

    bucket, _, blob = uri[len("gs://") :].partition("/")
    storage.Client(project=PROJECT).bucket(bucket).blob(blob).download_to_filename(str(dest))
    return dest


def main() -> None:
    from google.cloud import aiplatform

    args = parse_args()
    aiplatform.init(project=PROJECT, location=LOCATION, staging_bucket=STAGING, experiment=args.experiment)
    gold = pd.read_csv(args.gold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

    for config in args.configs:
        n_estimators, max_depth = (int(x) for x in config.split(":"))
        run_name = f"n{n_estimators}-d{max_depth}-{stamp}"
        model_dir = f"gs://{settings.ARTIFACTS_BUCKET}/models/exp-{run_name}"

        job = aiplatform.CustomJob(
            display_name=f"risk-train-{run_name}",
            worker_pool_specs=[
                {
                    "machine_spec": {"machine_type": args.machine_type},
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": args.image,
                        "args": [
                            "--data-uri", args.data_uri,
                            "--model-dir", model_dir,
                            "--n-estimators", str(n_estimators),
                            "--max-depth", str(max_depth),
                        ],
                    },
                }
            ],
            labels={"experiment": args.experiment},
        )
        job.run(service_account=SA, sync=True)
        print(f"{job.display_name}: {job.state.name}")

        with tempfile.TemporaryDirectory() as tmp:
            holdout = json.loads(read_gcs(f"{model_dir}/metrics.json", Path(tmp) / "m.json").read_text())
            model = joblib.load(read_gcs(f"{model_dir}/model.joblib", Path(tmp) / "model.joblib"))
        on_gold = gold_metrics(model, gold)

        with aiplatform.start_run(run_name):
            aiplatform.log_params(
                {"n_estimators": n_estimators, "max_depth": max_depth, "custom_job": job.resource_name}
            )
            aiplatform.log_metrics(
                {
                    **{f"holdout_{k}": round(holdout[k], 4) for k in ("accuracy", "precision", "recall", "f1")},
                    **{f"gold_{k}": round(on_gold[k], 4) for k in ("accuracy", "precision", "recall", "f1")},
                }
            )
        print(f"logged {run_name}: holdout_f1={holdout['f1']:.4f} gold_f1={on_gold['f1']:.4f}")

    print(aiplatform.get_experiment_df(args.experiment).to_string())


if __name__ == "__main__":
    main()
