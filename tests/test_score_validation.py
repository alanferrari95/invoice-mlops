import pytest
from fastapi.testclient import TestClient

from src.api.main import LoadedModel, PREDICTIONS_TABLE, create_app
from tests.conftest import MemorySink

VALID_PAYLOAD = {
    "amount": 900,
    "lines_sum": 700,
    "hour": 3,
    "is_new_vendor": 1,
    "vendor_avg_amount": 400,
    "amount_vs_avg_ratio": 2.25,
}


def test_score_rejects_negative_amount(client, sink):
    response = client.post("/score", json={**VALID_PAYLOAD, "amount": -10})
    assert response.status_code == 422
    assert sink.rows == []


def test_score_rejects_hour_out_of_range(client):
    response = client.post("/score", json={**VALID_PAYLOAD, "hour": 30})
    assert response.status_code == 422


def test_score_rejects_non_binary_new_vendor(client):
    response = client.post("/score", json={**VALID_PAYLOAD, "is_new_vendor": 2})
    assert response.status_code == 422


def test_valid_score_is_logged_once(client, sink):
    response = client.post("/score", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["risk"] in (0, 1)
    assert 0.0 <= body["score"] <= 1.0

    assert len(sink.rows) == 1
    table, row = sink.rows[0]
    assert table == PREDICTIONS_TABLE
    assert row["model_version"] == "test-1"
    assert row["score"] == body["score"]
    assert row["latency_ms"] >= 0


def test_logging_failure_does_not_break_scoring(tiny_model):
    app = create_app(
        model_loader=lambda: LoadedModel(tiny_model, version="test-1", uri="memory://tiny"),
        sink=MemorySink(fail=True),
        extractor_factory=lambda: None,
    )
    with TestClient(app) as c:
        assert c.post("/score", json=VALID_PAYLOAD).status_code == 200


def test_version_reports_loaded_model(client):
    body = client.get("/version").json()
    assert body["model_version"] == "test-1"
    assert body["model_uri"] == "memory://tiny"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_explain_returns_shap_values_that_add_up_to_the_margin(client, tiny_model):
    import math

    body = client.post("/score?explain=true", json=VALID_PAYLOAD).json()
    assert set(body["explanation"]) == {"amount", "lines_sum", "hour", "is_new_vendor",
                                        "vendor_avg_amount", "amount_vs_avg_ratio"}
    margin = body["base_value"] + sum(body["explanation"].values())
    assert 1 / (1 + math.exp(-margin)) == pytest.approx(body["score"], abs=1e-3)


def test_explanation_is_opt_in(client):
    assert client.post("/score", json=VALID_PAYLOAD).json()["explanation"] is None
