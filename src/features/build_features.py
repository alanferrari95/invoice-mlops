"""Build risk-model features from the raw invoice table in BigQuery.

The SQL in features.sql is the source of truth used by the pipeline;
compute_features() is the same logic in pandas, used by tests and local runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["invoice_id", "vendor_id", "amount", "lines_sum", "hour", "is_new_vendor", "risk"]
FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]
SQL_PATH = Path(__file__).with_name("features.sql")


def feature_query(source_table: str) -> str:
    return SQL_PATH.read_text(encoding="utf-8").format(source_table=source_table)


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    by_vendor = raw.groupby("vendor_id")["amount"]
    total, count = by_vendor.transform("sum"), by_vendor.transform("count")
    out = raw.copy()
    out["vendor_avg_amount"] = ((total - raw["amount"]) / (count - 1)).where(count > 1, raw["amount"])
    out["amount_vs_avg_ratio"] = out["amount"] / out["vendor_avg_amount"]
    return out[["invoice_id", *FEATURE_NAMES, "risk"]].sort_values("invoice_id").reset_index(drop=True)


def query_features(project: str, source_table: str, client=None) -> pd.DataFrame:
    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=project)
    return client.query(feature_query(source_table)).to_dataframe()


def main() -> None:
    parser = argparse.ArgumentParser()
    from src import settings

    parser.add_argument("--project", default=settings.PROJECT)
    parser.add_argument("--source-table", default=settings.INVOICES_TABLE)
    parser.add_argument("--out", default="data/features.csv")
    args = parser.parse_args()
    df = query_features(args.project, args.source_table)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
