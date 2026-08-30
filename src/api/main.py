"""FastAPI scoring service for invoice review risk."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models") / "model.joblib"
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


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Train a model first (e.g. python src/train/train.py)."
        )
    return joblib.load(MODEL_PATH)


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
    return {"app_version": APP_VERSION, "model_path": str(MODEL_PATH).replace("\\", "/")}


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
