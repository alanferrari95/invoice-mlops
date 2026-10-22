"""Deployment settings, all overridable with environment variables.

Defaults describe the reference deployment (project `invoice-mlops`, `europe-west4`). To run
the stack in your own project set GOOGLE_CLOUD_PROJECT (and REGION if needed); every bucket,
table, topic, image and Registry name below is derived from it.
"""

from __future__ import annotations

import os

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "invoice-mlops")
REGION = os.getenv("REGION", "europe-west4")

RAW_BUCKET = os.getenv("RAW_BUCKET", f"{PROJECT}-raw")
ARTIFACTS_BUCKET = os.getenv("ARTIFACTS_BUCKET", f"{PROJECT}-artifacts")
PIPELINE_ROOT = f"gs://{ARTIFACTS_BUCKET}/pipeline-root"
TEMPLATES = f"gs://{ARTIFACTS_BUCKET}/pipelines"

IMAGE_REPO = os.getenv("IMAGE_REPO", f"{REGION}-docker.pkg.dev/{PROJECT}/invoice-mlops")
WORK_SA = os.getenv("WORK_SA", f"invoice-mlops-work@{PROJECT}.iam.gserviceaccount.com")

DATASET = f"{PROJECT}.invoice_mlops"
INVOICES_TABLE = f"{DATASET}.invoices"
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", f"{DATASET}.predictions")
EXTRACTIONS_TABLE = os.getenv("EXTRACTIONS_TABLE", f"{DATASET}.extractions")

# Model Registry entries. Custom ids, so a fresh project gets the same resource names.
RISK_MODEL = os.getenv("RISK_MODEL", f"projects/{PROJECT}/locations/{REGION}/models/risk-classifier")
EXTRACTOR_MODEL = os.getenv(
    "EXTRACTOR_MODEL", f"projects/{PROJECT}/locations/{REGION}/models/invoice-extractor"
)

PIPELINE_ALERTS_TOPIC = f"projects/{PROJECT}/topics/invoice-mlops-pipeline-alerts"
DRIFT_ALERTS_TOPIC = f"projects/{PROJECT}/topics/invoice-mlops-drift-alerts"
MODEL_PROMOTED_TOPIC = f"projects/{PROJECT}/topics/invoice-mlops-model-promoted"
