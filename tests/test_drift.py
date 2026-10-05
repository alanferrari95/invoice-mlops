import numpy as np
import pandas as pd

from src.monitoring.drift import FEATURE_NAMES, compute_report, psi

rng = np.random.default_rng(0)


def _invoices(n, amount_scale=1.0):
    amount = rng.lognormal(6.5, 0.8, n) * amount_scale
    return pd.DataFrame({
        "amount": amount,
        "lines_sum": amount,
        "hour": rng.integers(8, 19, n),
        "is_new_vendor": rng.integers(0, 2, n),
        "vendor_avg_amount": amount / rng.uniform(0.8, 1.2, n),
        "amount_vs_avg_ratio": rng.uniform(0.8, 1.2, n),
    })


def test_psi_same_distribution_is_small():
    ref = rng.normal(0, 1, 5000)
    prod = rng.normal(0, 1, 5000)
    assert psi(ref, prod) < 0.1


def test_psi_shifted_distribution_is_large():
    ref = rng.normal(0, 1, 5000)
    prod = rng.normal(1.5, 1, 5000)
    assert psi(ref, prod) > 0.25


def test_psi_handles_binary_feature():
    ref = rng.integers(0, 2, 1000)
    assert psi(ref, rng.integers(0, 2, 1000)) < 0.1
    assert psi(ref, np.ones(1000)) > 0.25


def test_report_ok_when_no_shift():
    ref, prod = _invoices(400), _invoices(200)
    report = compute_report(ref, prod, rng.uniform(0, 1, 400), rng.uniform(0, 1, 200), 0.25, 0.01)
    assert report["status"] == "ok"
    assert set(report["features"]) == set(FEATURE_NAMES)


def test_report_flags_amount_and_score_shift():
    ref, prod = _invoices(400), _invoices(200, amount_scale=10)
    report = compute_report(ref, prod, rng.uniform(0, 0.3, 400), rng.uniform(0.7, 1, 200), 0.25, 0.01)
    assert report["status"] == "drift"
    assert "amount" in report["drifted"]
    assert "score" in report["drifted"]
