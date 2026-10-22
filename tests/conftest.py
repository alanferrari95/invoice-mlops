"""Shared fixtures. Nothing here talks to GCP: models are trained in memory
and BigQuery is replaced by an in-memory sink."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from src.api.main import FEATURE_NAMES, LoadedModel, create_app

DATA = Path(__file__).resolve().parents[1] / "data" / "invoices.csv"


class MemorySink:
    def __init__(self, fail: bool = False) -> None:
        self.rows: list[tuple[str, dict]] = []
        self.fail = fail

    def insert(self, table: str, rows: list[dict]) -> None:
        if self.fail:
            raise RuntimeError("BigQuery is down")
        self.rows.extend((table, r) for r in rows)


@pytest.fixture(scope="session")
def invoices() -> pd.DataFrame:
    return pd.read_csv(DATA)


@pytest.fixture(scope="session")
def tiny_model(invoices):
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=0, n_jobs=1)
    model.fit(invoices[FEATURE_NAMES], invoices["risk"])
    return model


@pytest.fixture
def sink() -> MemorySink:
    return MemorySink()


@pytest.fixture
def fake_gemini():
    from tests.fakes import FakeGeminiClient

    return FakeGeminiClient()


@pytest.fixture
def client(tiny_model, sink, fake_gemini):
    from src.extract.gemini import GeminiExtractor

    app = create_app(
        model_loader=lambda: LoadedModel(tiny_model, version="test-1", uri="memory://tiny"),
        sink=sink,
        extractor_factory=lambda: GeminiExtractor(fake_gemini, prompt_version="v2"),
    )
    with TestClient(app) as c:
        yield c
