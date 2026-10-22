"""FastAPI scoring service for invoice review risk."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

import joblib
import pandas as pd
import pandera.pandas as pa
from fastapi import FastAPI, HTTPException, Request
from pandera.errors import SchemaError
from pydantic import BaseModel

from src import registry
from src.registry import ModelFetch

APP_VERSION = "0.3.0"
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "invoice-mlops.invoice_mlops.predictions")

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


@dataclass
class LoadedModel:
    model: object
    version: str
    uri: str


class RowSink(Protocol):
    """Where prediction rows go. BigQuery in production, a list in tests."""

    def insert(self, table: str, rows: list[dict]) -> None: ...


class BigQuerySink:
    def __init__(self) -> None:
        self._client = None

    def insert(self, table: str, rows: list[dict]) -> None:
        from google.cloud import bigquery

        if self._client is None:
            self._client = bigquery.Client()
        errors = self._client.insert_rows_json(table, rows)
        if errors:
            logger.error("BigQuery insert_rows_json errors for %s: %s", table, errors)


def _download_gcs(uri: str, dest: Path) -> Path:
    from google.cloud import storage

    bucket_name, blob_name = uri[len("gs://") :].split("/", 1)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return dest


def load_model_from_uri(uri: str, version: str) -> LoadedModel:
    path = _download_gcs(uri, Path("/tmp/model.joblib")) if uri.startswith("gs://") else Path(uri)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path} (MODEL_URI={uri})")
    return LoadedModel(model=joblib.load(path), version=version, uri=uri)


def load_model(fetch: Optional[ModelFetch] = None) -> LoadedModel:
    """Production: a Model Registry version. Local/tests: MODEL_URI.

    Each Cloud Run revision is deployed with MODEL_VERSION_ID pinned (resolved from the
    `production` alias at deploy time), so a revision always serves the same model and a
    traffic rollback also rolls back the model. MODEL_ALIAS is the fallback reference.
    """
    registry_model = os.getenv("REGISTRY_MODEL")
    if registry_model:
        ref = os.getenv("MODEL_VERSION_ID") or os.getenv("MODEL_ALIAS", "production")
        version = registry.resolve(registry_model, ref, **({"fetch": fetch} if fetch else {}))
        if version is None:
            raise RuntimeError(f"No version of {registry_model} matches '{ref}'")
        logger.info("Serving %s@%s (version %s)", registry_model, ref, version.version_id)
        return load_model_from_uri(version.model_uri, version.label)
    uri = os.environ.get("MODEL_URI", "models/model.joblib")
    return load_model_from_uri(uri, os.getenv("MODEL_VERSION", "unversioned"))


def _positive_proba(model, X: pd.DataFrame) -> Optional[float]:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])


def create_app(
    model_loader: Callable[[], LoadedModel] = load_model,
    sink: Optional[RowSink] = None,
) -> FastAPI:
    """Build the app. Tests inject a fake loader and sink; production uses GCS/BigQuery."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loaded = model_loader()
        app.state.sink = sink if sink is not None else BigQuerySink()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/version")
    def version(request: Request) -> dict:
        loaded: LoadedModel = request.app.state.loaded
        return {
            "app_version": APP_VERSION,
            "model_version": loaded.version,
            "model_uri": loaded.uri,
            "revision": os.getenv("K_REVISION", "local"),
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(body: ScoreRequest, request: Request) -> ScoreResponse:
        row = {name: getattr(body, name) for name in FEATURE_NAMES}
        X = pd.DataFrame([row], columns=FEATURE_NAMES)
        try:
            X = ScoreSchema.validate(X)
        except SchemaError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

        loaded: LoadedModel = request.app.state.loaded
        t0 = time.perf_counter()
        risk = int(loaded.model.predict(X)[0])
        score_val = _positive_proba(loaded.model, X)
        latency_ms = (time.perf_counter() - t0) * 1000.0

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
            "model_version": loaded.version,
            "latency_ms": float(latency_ms),
        }
        try:
            request.app.state.sink.insert(PREDICTIONS_TABLE, [bq_row])
        except Exception:  # logging must never take the prediction down
            logger.exception("Failed to log prediction to %s", PREDICTIONS_TABLE)

        return ScoreResponse(risk=risk, score=score_val)

    return app


app = create_app()
