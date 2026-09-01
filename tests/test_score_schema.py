from pydantic import ValidationError
import pytest

from src.api.main import ScoreRequest


def test_valid_payload():
    ScoreRequest(
        amount=900,
        lines_sum=700,
        hour=3,
        is_new_vendor=1,
        vendor_avg_amount=400,
        amount_vs_avg_ratio=2.25,
    )


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        ScoreRequest(amount=900)
