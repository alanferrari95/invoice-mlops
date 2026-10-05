"""Send realistic or deliberately skewed invoices to POST /score.

normal: rows sampled from the training CSV, as-is.
skewed: same rows, but amounts x8, night-time hours and only new vendors.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "amount",
    "lines_sum",
    "hour",
    "is_new_vendor",
    "vendor_avg_amount",
    "amount_vs_avg_ratio",
]


def skew(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["amount"] *= 8
    df["lines_sum"] = df["amount"]
    df["amount_vs_avg_ratio"] = df["amount"] / df["vendor_avg_amount"]
    df["hour"] = rng.integers(0, 6, len(df))
    df["is_new_vendor"] = 1
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="service base URL")
    parser.add_argument("--mode", choices=["normal", "skewed"], default="normal")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--data", default="data/invoices.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = pd.read_csv(args.data)[FEATURE_NAMES].sample(n=args.n, replace=True, random_state=args.seed)
    if args.mode == "skewed":
        rows = skew(rows, rng)

    ok = failed = 0
    for row in rows.to_dict(orient="records"):
        req = urllib.request.Request(
            f"{args.url.rstrip('/')}/score",
            data=json.dumps(row).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                ok += resp.status == 200
        except Exception as err:  # keep going; a cold start can 5xx once
            failed += 1
            print(f"request failed: {err}")
    print(f"mode={args.mode} sent={args.n} ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
