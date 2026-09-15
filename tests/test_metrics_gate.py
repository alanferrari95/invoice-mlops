import pytest

from src.eval.metrics_gate import assert_min_f1


def test_assert_min_f1_passes_at_threshold():
    assert assert_min_f1({"f1": 0.5}, 0.5) == "f1=0.5"


def test_assert_min_f1_fails_below_threshold():
    with pytest.raises(RuntimeError, match="below min_f1"):
        assert_min_f1({"f1": 0.49}, 0.5)


def test_assert_min_f1_fails_if_f1_missing():
    with pytest.raises(RuntimeError, match="missing key"):
        assert_min_f1({"accuracy": 0.9}, 0.5)
