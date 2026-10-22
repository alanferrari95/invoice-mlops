"""Generate a synthetic invoice dataset for risk labeling."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

N_VENDORS = 40
N_INVOICES = 400
SEED = 42
NIGHT_HOURS = {0, 1, 2, 3, 4, 5, 23}
NIGHT_FRAC = 0.12
LINES_MISMATCH_FRAC = 0.12  # ~88% match
LABEL_NOISE_FRAC = 0.08


def generate_dataset(
    n_invoices: int = N_INVOICES, seed: int = SEED, label_noise: float = LABEL_NOISE_FRAC
) -> pd.DataFrame:
    """Synthetic invoices with rule-bootstrapped labels.

    label_noise flips that fraction of labels (analyst mistakes). The training set
    uses 8%; the gold set uses 0, so it holds the labels the rules intend.
    """
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    vendors = []
    for vendor_id in range(1, N_VENDORS + 1):
        vendors.append(
            {
                "vendor_id": vendor_id,
                "vendor_name": fake.company(),
                "typical_amount": rng.uniform(50, 2000),
            }
        )

    vendor_ids = rng.integers(1, N_VENDORS + 1, size=n_invoices)
    vendor_lookup = {v["vendor_id"]: v for v in vendors}

    n_night = int(round(n_invoices * NIGHT_FRAC))
    is_night = np.zeros(n_invoices, dtype=bool)
    night_idx = rng.choice(n_invoices, size=n_night, replace=False)
    is_night[night_idx] = True

    hours = np.empty(n_invoices, dtype=int)
    day_hours = np.arange(8, 19)
    night_hour_list = np.array(sorted(NIGHT_HOURS))
    hours[is_night] = rng.choice(night_hour_list, size=n_night)
    hours[~is_night] = rng.choice(day_hours, size=n_invoices - n_night)

    n_mismatch = int(round(n_invoices * LINES_MISMATCH_FRAC))
    is_mismatch = np.zeros(n_invoices, dtype=bool)
    mismatch_idx = rng.choice(n_invoices, size=n_mismatch, replace=False)
    is_mismatch[mismatch_idx] = True

    rows = []
    seen_vendors: set[int] = set()
    for i in range(n_invoices):
        vid = int(vendor_ids[i])
        v = vendor_lookup[vid]
        amount = float(v["typical_amount"] * rng.lognormal(mean=0.0, sigma=0.25))
        if is_mismatch[i]:
            delta = float(rng.uniform(5, 80))
            sign = rng.choice([-1.0, 1.0])
            lines_sum = amount + sign * delta
        else:
            lines_sum = amount
        is_new = 1 if vid not in seen_vendors else 0
        seen_vendors.add(vid)
        rows.append(
            {
                "invoice_id": i + 1,
                "vendor_id": vid,
                "vendor_name": v["vendor_name"],
                "amount": amount,
                "lines_sum": lines_sum,
                "hour": int(hours[i]),
                "is_new_vendor": is_new,
            }
        )

    df = pd.DataFrame(rows)

    vendor_avg = []
    for idx, row in df.iterrows():
        others = df.loc[(df["vendor_id"] == row["vendor_id"]) & (df.index != idx), "amount"]
        if len(others) == 0:
            vendor_avg.append(float(row["amount"]))
        else:
            vendor_avg.append(float(others.mean()))
    df["vendor_avg_amount"] = vendor_avg
    df["amount_vs_avg_ratio"] = df["amount"] / df["vendor_avg_amount"]

    risk = (
        (df["amount_vs_avg_ratio"] > 2.5)
        | ((df["amount"] - df["lines_sum"]).abs() > 1.0)
        | (df["is_new_vendor"] == 1)
        | (df["hour"] < 6)
        | (df["hour"] > 22)
    ).astype(int)

    n_flip = int(round(n_invoices * label_noise))
    flip_idx = rng.choice(df.index.to_numpy(), size=n_flip, replace=False)
    risk = risk.copy()
    risk.loc[flip_idx] = 1 - risk.loc[flip_idx]
    df["risk"] = risk

    cols = [
        "invoice_id",
        "vendor_id",
        "vendor_name",
        "amount",
        "lines_sum",
        "hour",
        "is_new_vendor",
        "vendor_avg_amount",
        "amount_vs_avg_ratio",
        "risk",
    ]
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_INVOICES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--label-noise", type=float, default=LABEL_NOISE_FRAC)
    parser.add_argument("--out", default="data/invoices.csv")
    args = parser.parse_args()

    df = generate_dataset(args.n, args.seed, args.label_noise)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"shape: {df.shape}")
    print(f"risk rate: {df['risk'].mean():.4f}")
    print(df["risk"].value_counts())


if __name__ == "__main__":
    main()
