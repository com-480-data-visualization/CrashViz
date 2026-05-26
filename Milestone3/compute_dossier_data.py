"""
compute_dossier_data.py

Reads market_data_2000_2025.csv and produces two JSON files
consumed by the dossier.html inner page:

  1. price_series.json
     Full daily price series (all ~6,410 rows) per asset.
     Used by D3 to draw the frozen line chart.

  2. dossier_timeseries.json
     Per-asset, per-month-end snapshot of derived risk metrics:
       - price_raw, price_norm  (closing price + base-100)
       - vol_30d/63d/252d       (annualised rolling volatility, %)
       - drawdown_current       (% from last all-time high to this date)
       - drawdown_max           (worst ever drawdown from inception to this date)

Usage:
  python compute_dossier_data.py
  python compute_dossier_data.py --csv path/to/data.csv --out-dir ./data

Navigation contract (matches index.html sessionStorage keys):
  sessionStorage["mg_currentMonth"]  →  int month index  →  year=2000+floor(idx/12), month=idx%12
  sessionStorage["mg_lastAsset"]     →  globe_id string  →  mapped to CSV column via GLOBE_ID_TO_COL
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  (must stay in sync with index.html)
# ─────────────────────────────────────────────────────────────────────────────

# Column order in the CSV (= asset index 0-7 in the force graph)
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

# Human-readable labels (matches LABELS array in index.html)
LABELS = [
    "CRUDE OIL",
    "USD INDEX",
    "GOLD",
    "COPPER",
    "SILVER",
    "CORN",
    "WHEAT",
    "S&P 500",
]

# Short names (matches SHORT array in index.html)
SHORT = ["OIL", "USD", "GOLD", "COPPER", "SILVER", "CORN", "WHEAT", "S&P 500"]

# Ticker symbols (matches TICKERS array in index.html)
TICKERS = ["CL=F", "DX-Y.NYB", "GC=F", "HG=F", "SI=F", "ZC=F", "ZW=F", "^GSPC"]

# Neon colors (matches COLORS array in index.html)
COLORS = [
    "#FF8C1A",  # Crude Oil   – amber
    "#5BB0FF",  # USD Index   – blue
    "#FFD93D",  # Gold        – yellow
    "#FF6B3D",  # Copper      – orange
    "#D6D6D6",  # Silver      – grey
    "#9CE53F",  # Corn        – lime
    "#3BD9A6",  # Wheat       – teal
    "#B8A4FF",  # S&P 500     – lavender
]

# Globe page hash IDs → CSV column (matches GLOBE_IDS in index.html)
GLOBE_ID_TO_COL = {
    "crude_oil":  "Crude_Oil",
    "usd_index":  "US_Dollar_Index",
    "gold":       "Gold",
    "copper":     "Copper",
    "silver":     "Silver",
    "corn":       "Corn",
    "wheat":      "Wheat",
    "sp500":      "SP500",
}

# Rolling volatility windows (in trading days)
VOL_WINDOWS = {"vol_30d": 30, "vol_63d": 63, "vol_252d": 252}

# Annualisation factor
TRADING_DAYS_PER_YEAR = 252

DEFAULT_CSV     = "market_data_2000_2025.csv"
DEFAULT_OUT_DIR = "."


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute dossier JSON files from market price CSV."
    )
    p.add_argument("--csv",     default=DEFAULT_CSV,     help="Path to market_data_2000_2025.csv")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for output JSON files")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    missing = [a for a in ASSETS if a not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df[ASSETS].sort_index()


# ─────────────────────────────────────────────────────────────────────────────
#  DERIVED METRICS
# ─────────────────────────────────────────────────────────────────────────────

def base100_normalize(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalise each column to 100 at the first valid observation."""
    first_valid = prices.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else 1)
    return (prices / first_valid) * 100


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def rolling_annualised_vol(log_ret: pd.DataFrame, window: int, min_periods: int = 10) -> pd.DataFrame:
    """Annualised volatility from log returns: std × sqrt(252)."""
    return log_ret.rolling(window=window, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_drawdown_series(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      drawdown_current  – (price / running_max - 1) at each date  [0, -1]
      drawdown_max      – worst (most negative) drawdown from inception to each date [0, -1]
    """
    running_max = prices.cummax()
    dd_current  = (prices / running_max) - 1               # 0 at new highs, negative in drawdown
    dd_max      = dd_current.cummin()                       # most negative seen so far
    return dd_current, dd_max


def compute_peak_trough_dates(
    prices:      pd.DataFrame,
    dd_current:  pd.DataFrame,
    sample_dates: pd.DatetimeIndex,
) -> tuple[dict, dict]:
    """
    For each asset and each month-end sample date, find:
      peak_date   – date of the price peak that preceded the worst drawdown
                    (up to and including the sample date)
      trough_date – date of the worst drawdown bottom
                    (up to and including the sample date)

    Uses the pre-computed dd_current series to locate the trough efficiently,
    then scans prices[:trough] to find the preceding peak.
    Only iterates over ~300 month-end dates × 8 assets (not 6,410 rows).
    """
    peak_dates   = {col: {} for col in prices.columns}
    trough_dates = {col: {} for col in prices.columns}

    for col in prices.columns:
        price_col = prices[col].dropna()
        dd_col    = dd_current[col].dropna()

        for date in sample_dates:
            # Slice both series up to this month-end date
            dd_window = dd_col[dd_col.index <= date]
            if dd_window.empty:
                continue

            # Trough = day with the most negative drawdown up to frozen date
            trough_date = dd_window.idxmin()

            # Peak = highest price on or before the trough date
            price_window = price_col[price_col.index <= trough_date]
            if price_window.empty:
                continue
            peak_date = price_window.idxmax()

            key = date.strftime("%Y-%m")
            peak_dates[col][key]   = peak_date.strftime("%Y-%m-%d")
            trough_dates[col][key] = trough_date.strftime("%Y-%m-%d")

    return peak_dates, trough_dates


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT 1: price_series.json
#  Full daily series – all ~6,410 rows – used by D3 line chart
# ─────────────────────────────────────────────────────────────────────────────

def build_price_series(prices: pd.DataFrame, norm: pd.DataFrame) -> dict:
    """
    Schema:
    {
      "meta": { "assets": [...], "labels": [...], ... },
      "series": {
        "Crude_Oil": [
          { "date": "2000-08-30", "price": 32.5, "norm": 100.0 },
          ...
        ],
        ...
      }
    }
    """
    series = {}
    dates = prices.index.strftime("%Y-%m-%d").tolist()

    for col in ASSETS:
        raw_vals  = prices[col].tolist()
        norm_vals = norm[col].tolist()
        series[col] = [
            {
                "date":  d,
                "price": round(float(r), 4) if not np.isnan(r) else None,
                "norm":  round(float(n), 4) if not np.isnan(n) else None,
            }
            for d, r, n in zip(dates, raw_vals, norm_vals)
        ]

    return {
        "meta": {
            "assets":        ASSETS,
            "labels":        LABELS,
            "short":         SHORT,
            "tickers":       TICKERS,
            "colors":        COLORS,
            "globe_id_map":  GLOBE_ID_TO_COL,
            "total_rows":    len(prices),
            "date_start":    prices.index[0].strftime("%Y-%m-%d"),
            "date_end":      prices.index[-1].strftime("%Y-%m-%d"),
            "note":          "Full daily price series. norm=base-100 from first valid observation.",
        },
        "series": series,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT 2: dossier_timeseries.json
#  Monthly snapshots of risk metrics – indexed by YYYY-MM (month-end)
# ─────────────────────────────────────────────────────────────────────────────

def build_dossier_timeseries(
    prices:       pd.DataFrame,
    norm:         pd.DataFrame,
    log_ret:      pd.DataFrame,
    dd_current:   pd.DataFrame,
    dd_max:       pd.DataFrame,
    vol_frames:   dict[str, pd.DataFrame],
    peak_dates:   dict,
    trough_dates: dict,
) -> dict:
    """
    Schema:
    {
      "meta": { ... },
      "assets": ["Crude_Oil", ...],
      "dates":  ["2000-09", "2000-10", ...],
      "timeseries": {
        "Crude_Oil": {
          "2000-09": {
            "price":                33.88,
            "norm":                 104.23,
            "vol_30d":              0.3512,   // annualised, e.g. 0.35 = 35%
            "vol_63d":              0.3210,
            "vol_252d":             0.2984,
            "drawdown_current":    -0.0412,   // e.g. -0.04 = -4% from peak
            "drawdown_max":        -0.1823,   // worst ever up to this date
            "drawdown_peak_date":  "2000-08-30",  // peak before worst drawdown
            "drawdown_trough_date":"2000-09-28"   // bottom of worst drawdown
          },
          ...
        },
        ...
      }
    }
    """
    # Month-end sample dates = last trading day of each calendar month
    sample_dates = pd.Series(prices.index, index=prices.index).resample("ME").last().dropna()

    timeseries = {col: {} for col in ASSETS}

    for date in sample_dates:
        key = date.strftime("%Y-%m")

        for col in ASSETS:
            row = {}

            # Price + norm
            p = prices.loc[date, col]
            n = norm.loc[date, col]
            row["price"] = round(float(p), 4) if not np.isnan(p) else None
            row["norm"]  = round(float(n), 4) if not np.isnan(n) else None

            # Rolling volatilities
            for vol_key, vol_df in vol_frames.items():
                v = vol_df.loc[date, col] if date in vol_df.index else float("nan")
                row[vol_key] = round(float(v), 6) if not np.isnan(v) else None

            # Drawdown scalar metrics
            dc = dd_current.loc[date, col] if date in dd_current.index else float("nan")
            dm = dd_max.loc[date, col]     if date in dd_max.index     else float("nan")
            row["drawdown_current"] = round(float(dc), 6) if not np.isnan(dc) else None
            row["drawdown_max"]     = round(float(dm), 6) if not np.isnan(dm) else None

            # Drawdown zoom anchor dates
            row["drawdown_peak_date"]   = peak_dates.get(col, {}).get(key, None)
            row["drawdown_trough_date"] = trough_dates.get(col, {}).get(key, None)

            timeseries[col][key] = row

    dates_list = sorted({k for col in ASSETS for k in timeseries[col]})

    return {
        "meta": {
            "assets":       ASSETS,
            "labels":       LABELS,
            "short":        SHORT,
            "tickers":      TICKERS,
            "colors":       COLORS,
            "globe_id_map": GLOBE_ID_TO_COL,
            "vol_windows":  VOL_WINDOWS,
            "sampled":      "month-end (last trading day of each calendar month)",
            "vol_unit":     "annualised fraction (e.g. 0.35 = 35%)",
            "dd_unit":      "fraction (e.g. -0.12 = -12% from peak)",
            "session_keys": {
                "month_index": "mg_currentMonth",
                "asset_id":    "mg_lastAsset",
            },
            "month_index_formula": "year = 2000 + floor(idx/12) ; month = idx % 12 ; key = YYYY-MM",
        },
        "assets":      ASSETS,
        "dates":       dates_list,
        "timeseries":  timeseries,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WRITE JSON  (compact – no extra whitespace)
# ─────────────────────────────────────────────────────────────────────────────

def write_json(obj: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    size_kb = path.stat().st_size / 1024
    print(f"  Written: {path}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading prices from : {args.csv}")
    prices = load_prices(args.csv)
    print(f"  Rows loaded       : {len(prices)}")
    print(f"  Date range        : {prices.index[0].date()} → {prices.index[-1].date()}")

    # Derived series
    print("\nComputing derived metrics …")
    norm        = base100_normalize(prices)
    log_ret     = log_returns(prices)
    dd_current, dd_max = compute_drawdown_series(prices)

    vol_frames = {
        vol_key: rolling_annualised_vol(log_ret, window=w)
        for vol_key, w in VOL_WINDOWS.items()
    }

    # Peak / trough dates for drawdown zoom interaction
    print("Computing drawdown peak & trough dates …")
    sample_dates = pd.Series(prices.index, index=prices.index).resample("ME").last().dropna()
    peak_dates, trough_dates = compute_peak_trough_dates(prices, dd_current, sample_dates)

    # Build & write outputs
    print("\nBuilding price_series.json …")
    ps_obj  = build_price_series(prices, norm)
    write_json(ps_obj, out_dir / "price_series.json")

    print("Building dossier_timeseries.json …")
    dt_obj  = build_dossier_timeseries(
        prices, norm, log_ret, dd_current, dd_max,
        vol_frames, peak_dates, trough_dates,
    )
    write_json(dt_obj, out_dir / "dossier_timeseries.json")

    # Sanity checks
    print("\n── Sanity checks ──────────────────────────────────────────")
    print(f"  Total month snapshots : {len(dt_obj['dates'])}")
    print(f"  First / last month    : {dt_obj['dates'][0]} / {dt_obj['dates'][-1]}")

    sample_key = "2008-10"
    if sample_key in dt_obj["timeseries"]["Gold"]:
        g = dt_obj["timeseries"]["Gold"][sample_key]
        print(f"\n  Gold @ {sample_key}:")
        print(f"    price               = {g['price']}")
        print(f"    norm                = {g['norm']:.2f}")
        print(f"    vol_63d             = {g['vol_63d']:.4f}  (≈ {g['vol_63d']*100:.1f}%)")
        print(f"    drawdown_curr       = {g['drawdown_current']:.4f}  (≈ {g['drawdown_current']*100:.1f}%)")
        print(f"    drawdown_max        = {g['drawdown_max']:.4f}  (≈ {g['drawdown_max']*100:.1f}%)")
        print(f"    drawdown_peak_date  = {g['drawdown_peak_date']}")
        print(f"    drawdown_trough_date= {g['drawdown_trough_date']}")

    sp = dt_obj["timeseries"]["SP500"].get("2009-03", {})
    if sp:
        print(f"\n  S&P 500 @ 2009-03 (GFC trough):")
        print(f"    drawdown_curr       = {sp['drawdown_current']:.4f}  (≈ {sp['drawdown_current']*100:.1f}%)")
        print(f"    drawdown_max        = {sp['drawdown_max']:.4f}  (≈ {sp['drawdown_max']*100:.1f}%)")
        print(f"    drawdown_peak_date  = {sp['drawdown_peak_date']}")
        print(f"    drawdown_trough_date= {sp['drawdown_trough_date']}")

    print("\nDone.")


if __name__ == "__main__":
    main()