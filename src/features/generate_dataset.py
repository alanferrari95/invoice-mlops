"""Generate a synthetic invoice dataset for risk labeling."""

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


def generate_dataset() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(SEED)
    rng = np.random.default_rng(SEED)

    vendors = []
    for vendor_id in range(1, N_VENDORS + 1):
        vendors.append(
            {
                "vendor_id": vendor_id,
                "vendor_name": fake.company(),
                "typical_amount": rng.uniform(50, 2000),
            }
        )

    vendor_ids = rng.integers(1, N_VENDORS + 1, size=N_INVOICES)
    vendor_lookup = {v["vendor_id"]: v for v in vendors}

    n_night = int(round(N_INVOICES * NIGHT_FRAC))
    is_night = np.zeros(N_INVOICES, dtype=bool)
    night_idx = rng.choice(N_INVOICES, size=n_night, replace=False)
    is_night[night_idx] = True

    hours = np.empty(N_INVOICES, dtype=int)
    day_hours = np.arange(8, 19)
    night_hour_list = np.array(sorted(NIGHT_HOURS))
    hours[is_night] = rng.choice(night_hour_list, size=n_night)
    hours[~is_night] = rng.choice(day_hours, size=N_INVOICES - n_night)

    n_mismatch = int(round(N_INVOICES * LINES_MISMATCH_FRAC))
    is_mismatch = np.zeros(N_INVOICES, dtype=bool)
    mismatch_idx = rng.choice(N_INVOICES, size=n_mismatch, replace=False)
    is_mismatch[mismatch_idx] = True

    rows = []
    seen_vendors: set[int] = set()
    for i in range(N_INVOICES):
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

    n_flip = int(round(N_INVOICES * LABEL_NOISE_FRAC))
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
    df = generate_dataset()
    out_path = Path("data") / "invoices.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"shape: {df.shape}")
    print(f"risk rate: {df['risk'].mean():.4f}")
    print(df["risk"].value_counts())


if __name__ == "__main__":
    main()
