import json

from src.train.train import fit, save


def test_fit_returns_model_and_holdout_metrics(invoices):
    model, metrics = fit(invoices, n_estimators=20, max_depth=3)
    assert metrics["n_train"] + metrics["n_test"] == len(invoices)
    assert 0.0 <= metrics["f1"] <= 1.0
    assert hasattr(model, "predict_proba")


def test_save_writes_model_and_metrics(invoices, tmp_path):
    model, metrics = fit(invoices, n_estimators=10, max_depth=2)
    save(model, metrics, tmp_path / "out")
    assert (tmp_path / "out" / "model.joblib").is_file()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["f1"] == metrics["f1"]
