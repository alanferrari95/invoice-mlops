"""FastAPI scoring service for invoice review risk."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel

MODEL_URI = os.getenv("MODEL_URI", "models/model.joblib")
MODEL_PATH = Path("/tmp/model.joblib") if MODEL_URI.startswith("gs://") else Path(MODEL_URI)
APP_VERSION = "0.1.0"

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]


class ScoreRequest(BaseModel):
    amount: float
    lines_sum: float
    hour: float
    is_new_vendor: float
    vendor_avg_amount: float
    amount_vs_avg_ratio: float


class ScoreResponse(BaseModel):
    risk: int
    score: Optional[float]


# def load_model():
#     if MODEL_URI.startswith("gs://"):
#         # gs://bucket/blob — bucket, then blob name after the first /
#         bucket_name, blob_name = MODEL_URI[len("gs://"):].split("/", 1)
#         blob = storage.Client().bucket(bucket_name).blob(blob_name)
#         blob.download_to_filename(str(MODEL_PATH))

#     if not MODEL_PATH.is_file():
#         raise FileNotFoundError(
#             f"Model file not found: {MODEL_PATH}. "
#             "Train a model first (e.g. python src/train/train.py)."
#         )
#     return joblib.load(MODEL_PATH)

def load_model():
    uri = os.environ.get("MODEL_URI", "models/model.joblib")
    if uri.startswith("gs://"):
        dest = Path("/tmp/model.joblib")
        bucket_name, blob_name = uri[len("gs://") :].split("/", 1)
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
        path = dest
    else:
        path = Path(uri)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path} (MODEL_URI={uri})")
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    return {
        "app_version": APP_VERSION,
        "model_path": str(MODEL_PATH).replace("\\", "/"),
        "model_uri": MODEL_URI,
    }


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest) -> ScoreResponse:
    model = app.state.model
    row = {name: getattr(body, name) for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    risk = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # class 1 probability
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            score_val = float(proba[classes.index(1)])
        else:
            score_val = float(proba[-1])
    else:
        score_val = None
    return ScoreResponse(risk=risk, score=score_val)
