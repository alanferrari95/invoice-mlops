from src.eval.gate import decide, gold_metrics


def test_rejects_below_absolute_threshold():
    d = decide(0.70, None, min_f1=0.75, max_regression=0.02)
    assert not d.promote and "below min_f1" in d.reason


def test_promotes_first_model_without_champion():
    d = decide(0.80, None, min_f1=0.75, max_regression=0.02)
    assert d.promote and "no comparable champion" in d.reason


def test_rejects_regression_against_champion():
    d = decide(0.80, 0.86, min_f1=0.75, max_regression=0.02)
    assert not d.promote and "regresses" in d.reason


def test_tolerates_small_regression():
    assert decide(0.85, 0.86, min_f1=0.75, max_regression=0.02).promote


def test_impossible_threshold_always_rejects():
    assert not decide(0.99, None, min_f1=1.01, max_regression=0.0).promote


def test_gold_metrics_on_gold_set(tiny_model):
    import pandas as pd

    gold = pd.read_csv("data/gold/risk_gold.csv")
    m = gold_metrics(tiny_model, gold)
    assert m["n_gold"] == len(gold)
    assert 0.0 <= m["f1"] <= 1.0
