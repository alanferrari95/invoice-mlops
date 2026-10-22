import numpy as np

from src.features.build_features import FEATURE_NAMES, RAW_COLUMNS, compute_features, feature_query


def test_pandas_features_match_the_committed_dataset(invoices):
    """compute_features (the pandas twin of features.sql) reproduces data/invoices.csv."""
    out = compute_features(invoices[RAW_COLUMNS])
    for col in ("vendor_avg_amount", "amount_vs_avg_ratio"):
        np.testing.assert_allclose(out[col], invoices[col], rtol=1e-9)
    assert list(out.columns) == ["invoice_id", *FEATURE_NAMES, "risk"]


def test_single_invoice_vendor_uses_own_amount(invoices):
    raw = invoices[RAW_COLUMNS].head(1)
    out = compute_features(raw)
    assert out.loc[0, "vendor_avg_amount"] == raw.iloc[0]["amount"]
    assert out.loc[0, "amount_vs_avg_ratio"] == 1.0


def test_query_targets_the_given_table():
    sql = feature_query("p.d.invoices")
    assert "`p.d.invoices`" in sql
    assert "PARTITION BY vendor_id" in sql


def test_query_features_uses_injected_client(invoices):
    class FakeJob:
        def to_dataframe(self):
            return invoices

    class FakeClient:
        def __init__(self):
            self.sql = None

        def query(self, sql):
            self.sql = sql
            return FakeJob()

    from src.features.build_features import query_features

    fake = FakeClient()
    assert len(query_features("p", "p.d.t", client=fake)) == len(invoices)
    assert "`p.d.t`" in fake.sql
