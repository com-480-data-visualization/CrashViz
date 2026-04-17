"""
compute_correlations.py

Reads market_data_2000_2025.csv, computes 63-day rolling Pearson correlations
for all asset pairs, samples at month-end, and writes correlation_data.json.

The full matrix lets D3 index directly as matrix[i][j] with no pair index arithmetic required.

Usage:
  python compute_correlations.py
  python compute_correlations.py --csv path/to/data.csv --window 21 --out out.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Config

ASSETS = [
    "Crude_Oil",
    "US_Dollar_Index",
    "Gold",
    "Copper",
    "Silver",
    "Corn",
    "Wheat",
    "SP500",
]

DEFAULT_CSV    = "market_data_2000_2025.csv"
DEFAULT_WINDOW = 63   # ~3 months of trading days
DEFAULT_OUT    = "correlation_data.json"
MIN_PERIODS    = 20   # minimum observations required to emit a matrix

# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Compute rolling correlation matrices for the D3 dashboard.")
    p.add_argument("--csv",    default=DEFAULT_CSV,    help="Path to cleaned price CSV")
    p.add_argument("--window", default=DEFAULT_WINDOW, type=int, help="Rolling window in trading days")
    p.add_argument("--out",    default=DEFAULT_OUT,    help="Output JSON path")
    return p.parse_args()

# Core computation

def load_prices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    missing = [a for a in ASSETS if a not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df[ASSETS].sort_index()

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns are additive and better-behaved for rolling correlation."""
    return np.log(prices / prices.shift(1)).dropna()

def rolling_corr_matrix(
    log_ret: pd.DataFrame,
    window: int,
    min_periods: int,
    sample_dates: pd.DatetimeIndex,
) -> dict[str, list]:
    """
    For each date in sample_dates, compute the Pearson correlation matrix
    over the preceding `window` trading-day log-returns.
    """
    matrices = {}

    for date in sample_dates:
        # Slice the window ending at this date
        window_data = log_ret[log_ret.index <= date].tail(window)

        if len(window_data) < min_periods:
            continue

        corr = window_data.corr()                    # 8x8 DataFrame
        key  = date.strftime("%Y-%m")

        # Store as nested list, rounded to 4 d.p.
        matrices[key] = [
            [round(float(corr.iloc[i, j]), 4) for j in range(len(ASSETS))]
            for i in range(len(ASSETS))
        ]

    return matrices

# Entry point

def main():
    args = parse_args()

    print(f"Loading prices from  : {args.csv}")
    prices  = load_prices(args.csv)
    log_ret = compute_log_returns(prices)

    # Month-end sample dates (last trading day of each calendar month)
    sample_dates = log_ret.resample("ME").last().index
    print(f"Rolling window       : {args.window} trading days")
    print(f"Month-end samples    : {len(sample_dates)} dates "
          f"({sample_dates[0].strftime('%Y-%m')} -> {sample_dates[-1].strftime('%Y-%m')})")

    matrices = rolling_corr_matrix(log_ret, args.window, MIN_PERIODS, sample_dates)
    dates    = sorted(matrices.keys())

    output = {
        "meta": {
            "assets":  ASSETS,
            "window":  args.window,
            "sampled": "month-end",
            "source":  Path(args.csv).name,
        },
        "dates":    dates,
        "matrices": matrices,
    }

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))   # compact, no extra whitespace

    size_kb = out_path.stat().st_size / 1024
    print(f"Written              : {out_path}  ({size_kb:.1f} KB, {len(dates)} matrices)")

    # Quick sanity check
    sample_key = "2008-10"
    if sample_key in matrices:
        print(f"\nSanity check - {sample_key} diagonal (should all be 1.0):")
        diag = [matrices[sample_key][i][i] for i in range(len(ASSETS))]
        print("  ", diag)
        print(f"  Gold-Silver correlation : {matrices[sample_key][2][4]:.4f}  (expected ≈ 0.6)")


if __name__ == "__main__":
    main()
