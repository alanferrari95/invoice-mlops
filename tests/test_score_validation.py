from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("joblib")

from src.api.main import app

VALID_PAYLOAD = {
    "amount": 900,
    "lines_sum": 700,
    "hour": 3,
    "is_new_vendor": 1,
    "vendor_avg_amount": 400,
    "amount_vs_avg_ratio": 2.25,
}


def _client() -> TestClient:
    if not Path("models/model.joblib").is_file():
        pytest.skip("models/model.joblib is missing")
    return TestClient(app)


def test_score_rejects_negative_amount():
    payload = {**VALID_PAYLOAD, "amount": -10}
    response = _client().post("/score", json=payload)
    assert response.status_code == 422


def test_score_rejects_hour_out_of_range():
    payload = {**VALID_PAYLOAD, "hour": 30}
    response = _client().post("/score", json=payload)
    assert response.status_code == 422
