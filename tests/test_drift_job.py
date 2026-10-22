"""End-to-end run of the drift job's main() with BigQuery, GCS, Pub/Sub and Vertex faked."""

import json

import pytest

from src.monitoring import drift


@pytest.fixture
def wired(monkeypatch, invoices, tiny_model):
    calls = {"published": [], "retrained": 0, "versions": []}

    def production(table, hours, model_version=None):
        calls["versions"].append(model_version)
        return calls["prod"]

    monkeypatch.setattr(drift, "load_production", production)
    monkeypatch.setattr(drift, "load_reference", lambda uri: invoices)
    monkeypatch.setattr(drift, "load_model", lambda uri: tiny_model)
    monkeypatch.setattr(drift, "publish", lambda topic, payload: calls["published"].append(payload) or "msg-1")

    def fake_retrain():
        calls["retrained"] += 1
        return {"submitted": True, "reason": "submitted (drift)", "job": "pipelineJobs/x"}

    monkeypatch.setattr(drift, "retrain", fake_retrain)
    return calls


def _scored(df, model):
    out = df[drift.FEATURE_NAMES].copy()
    out["score"] = model.predict_proba(out)[:, 1]
    return out


def test_normal_traffic_is_ok(wired, invoices, tiny_model, capsys):
    wired["prod"] = _scored(invoices.sample(200, random_state=1), tiny_model)
    drift.main(["--topic", "projects/p/topics/t", "--retrain"])
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["status"] == "ok"
    assert wired["published"] == [] and wired["retrained"] == 0


def test_skewed_traffic_alerts_and_retrains(wired, invoices, tiny_model, capsys):
    skewed = invoices.sample(200, random_state=1).copy()
    skewed["amount"] *= 8
    skewed["lines_sum"] = skewed["amount"]
    skewed["amount_vs_avg_ratio"] = skewed["amount"] / skewed["vendor_avg_amount"]
    skewed["hour"] = 3
    skewed["is_new_vendor"] = 1
    wired["prod"] = _scored(skewed, tiny_model)

    drift.main(["--topic", "projects/p/topics/t", "--retrain"])
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["status"] == "drift"
    assert {"amount", "hour", "is_new_vendor", "score"} <= set(report["drifted"])
    assert report["retrain"]["submitted"] is True
    assert len(wired["published"]) == 1 and wired["retrained"] == 1


def test_drift_without_retrain_flag_only_alerts(wired, invoices, tiny_model):
    skewed = invoices.sample(200, random_state=1).copy()
    skewed["hour"] = 3
    wired["prod"] = _scored(skewed, tiny_model)
    drift.main(["--topic", "projects/p/topics/t", "--no-retrain"])
    assert len(wired["published"]) == 1 and wired["retrained"] == 0


def test_too_few_rows_is_not_judged(wired, invoices, tiny_model, capsys):
    wired["prod"] = _scored(invoices.head(10), tiny_model)
    drift.main(["--topic", "projects/p/topics/t", "--retrain"])
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_data"
    assert wired["published"] == []


def test_registry_version_filters_production_rows(wired, invoices, tiny_model, monkeypatch):
    from src import registry

    monkeypatch.setattr(
        registry, "resolve", lambda model, ref: registry.RegistryVersion(model, "5", "gs://b/m")
    )
    wired["prod"] = _scored(invoices.sample(100, random_state=2), tiny_model)
    drift.main(["--registry-model", "projects/p/locations/europe-west4/models/1"])
    assert wired["versions"] == ["v5"]
