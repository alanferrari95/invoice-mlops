"""FastAPI scoring service for invoice review risk."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import pandera as pa
from fastapi import FastAPI, HTTPException
from google.cloud import bigquery, storage
from pandera.errors import SchemaError
from pydantic import BaseModel

MODEL_URI = os.getenv("MODEL_URI", "models/model.joblib")
MODEL_PATH = Path("/tmp/model.joblib") if MODEL_URI.startswith("gs://") else Path(MODEL_URI)
APP_VERSION = "0.1.0"
PREDICTIONS_TABLE = "invoice-mlops.invoice_mlops.predictions"

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]

ScoreSchema = pa.DataFrameSchema(
    {
        "amount": pa.Column(float, pa.Check.gt(0), coerce=True),
        "lines_sum": pa.Column(float, pa.Check.gt(0), coerce=True),
        "hour": pa.Column(int, pa.Check.in_range(0, 23), coerce=True),
        "is_new_vendor": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        "vendor_avg_amount": pa.Column(float, pa.Check.gt(0), coerce=True),
        "amount_vs_avg_ratio": pa.Column(float, pa.Check.gt(0), coerce=True),
    },
    coerce=True,
)


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
    row = {name: getattr(body, name) for name in FEATURE_NAMES}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    try:
        X = ScoreSchema.validate(X)
    except SchemaError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    model = app.state.model
    
    t0 = time.perf_counter()
    risk = int(model.predict(X)[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

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

    feat = X.iloc[0]
    bq_row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "amount": float(feat["amount"]),
        "lines_sum": float(feat["lines_sum"]),
        "hour": int(feat["hour"]),
        "is_new_vendor": int(feat["is_new_vendor"]),
        "vendor_avg_amount": float(feat["vendor_avg_amount"]),
        "amount_vs_avg_ratio": float(feat["amount_vs_avg_ratio"]),
        "risk": risk,
        "score": score_val,
        "model_version": os.getenv("MODEL_VERSION", "0.1.0"),
        "latency_ms": float(latency_ms),
    }
    errors = bigquery.Client().insert_rows_json(PREDICTIONS_TABLE, [bq_row])
    if errors:
        logger.error("BigQuery insert_rows_json errors: %s", errors)

    return ScoreResponse(risk=risk, score=score_val)
