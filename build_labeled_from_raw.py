# build_labeled_from_raw.py
#
# Build labeled news dataset from per-ticker Finnhub CSVs:
# - Input: directory of CSVs with columns [ticker, published_utc, title, publisher, url]
# - Output: single CSV with forward returns & labels for 1d/3d/7d horizons

import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_daily_prices(ticker: str, start_date, end_date) -> pd.DataFrame:
    """
    Fetch daily adjusted Close prices for [start_date, end_date].
    Handles both flat and MultiIndex yfinance output.
    Returns columns: date, Close
    """
    print(f"  yfinance {ticker} {start_date} → {end_date}")
    prices = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),  # end exclusive
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if prices is None or prices.empty:
        print(f"  WARNING: no price data for {ticker}")
        return pd.DataFrame(columns=["date", "Close"])

    prices = prices.copy()

    # Handle both MultiIndex and flat columns
    if isinstance(prices.columns, pd.MultiIndex):
        level0 = prices.columns.get_level_values(0)
        if "Close" not in level0:
            print(f"  WARNING: no Close in MultiIndex for {ticker} (cols={list(prices.columns)})")
            return pd.DataFrame(columns=["date", "Close"])
        close_df = prices.xs("Close", axis=1, level=0)
        close_series = close_df.iloc[:, 0]
    else:
        if "Close" not in prices.columns:
            print(f"  WARNING: no Close in columns for {ticker} (cols={list(prices.columns)})")
            return pd.DataFrame(columns=["date", "Close"])
        close_series = prices["Close"]

    idx = prices.index
    date_values = pd.to_datetime(idx, errors="coerce").date

    out = pd.DataFrame({
        "date": date_values,
        "Close": close_series.values,
    })

    out = (
        out.dropna(subset=["date"])
           .drop_duplicates(subset=["date"])
           .sort_values("date")
           .reset_index(drop=True)
    )
    return out


def label_block(block: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For one ticker's headlines and its daily prices,
    compute forward 1/3/7-day returns & ±1% labels.
    """
    if prices.empty:
        for col in ["ret_1d", "ret_3d", "ret_7d", "label_1d", "label_3d", "label_7d"]:
            block[col] = np.nan
        return block

    dates = prices["date"].values
    close = prices["Close"].values

    ret_1d, ret_3d, ret_7d = [], [], []
    lab_1d, lab_3d, lab_7d = [], [], []

    def last_trading_index_on_or_before(ts):
        d = ts.date()
        idx = np.searchsorted(dates, d, side="right")
        j = idx - 1
        if j < 0:
            return None
        return j

    def ret_and_label(j0, horizon):
        j1 = j0 + horizon
        if j1 >= len(close):
            return np.nan, np.nan
        p0 = close[j0]
        p1 = close[j1]
        if p0 == 0 or np.isnan(p0) or np.isnan(p1):
            return np.nan, np.nan
        r = (p1 - p0) / p0
        # label: +1 if > +1%, -1 if < -1%, else 0
        if r > 0.01:
            lbl = 1
        elif r < -0.01:
            lbl = -1
        else:
            lbl = 0
        return r, lbl

    for ts in block["published_utc"]:
        j0 = last_trading_index_on_or_before(ts)
        if j0 is None:
            ret_1d.append(np.nan); ret_3d.append(np.nan); ret_7d.append(np.nan)
            lab_1d.append(np.nan); lab_3d.append(np.nan); lab_7d.append(np.nan)
            continue

        r1, l1 = ret_and_label(j0, 1)
        r3, l3 = ret_and_label(j0, 3)
        r7, l7 = ret_and_label(j0, 7)

        ret_1d.append(r1); lab_1d.append(l1)
        ret_3d.append(r3); lab_3d.append(l3)
        ret_7d.append(r7); lab_7d.append(l7)

    block["ret_1d"] = ret_1d
    block["ret_3d"] = ret_3d
    block["ret_7d"] = ret_7d
    block["label_1d"] = lab_1d
    block["label_3d"] = lab_3d
    block["label_7d"] = lab_7d
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory with per-ticker Finnhub CSVs")
    ap.add_argument("--outfile", default="news_labeled_alpha.csv", help="Output labeled CSV")
    args = ap.parse_args()

    indir = args.indir
    out_csv = args.outfile

    files = [f for f in os.listdir(indir) if f.lower().endswith(".csv")]
    if not files:
        raise SystemExit(f"No CSVs found in {indir}")

    print(f"Scanning {len(files)} ticker files from {indir}")

    blocks = []
    for i, fname in enumerate(files, 1):
        path = os.path.join(indir, fname)

        # Skip truly empty files (0 bytes)
        if os.path.getsize(path) == 0:
            print(f"[{i}/{len(files)}] {fname} is 0 bytes, skip")
            continue

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"[{i}/{len(files)}] {fname} has no columns / invalid CSV, skip")
            continue

        if df.empty:
            print(f"[{i}/{len(files)}] {fname} has no rows, skip")
            continue

        required = {"ticker", "published_utc", "title", "publisher", "url"}
        missing = required - set(df.columns)
        if missing:
            print(f"[{i}/{len(files)}] {fname} missing columns {missing}, skip")
            continue

        t = str(df["ticker"].iloc[0]).upper()
        df["ticker"] = t
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_utc"])
        if df.empty:
            print(f"[{i}/{len(files)}] {t} no valid timestamps, skip")
            continue

        min_ts = df["published_utc"].min()
        max_ts = df["published_utc"].max()

        start_date = (min_ts - pd.Timedelta(days=10)).date()
        end_date = (max_ts + pd.Timedelta(days=10)).date()

        print(f"\n[{i}/{len(files)}] {t}: headlines={len(df)}, price range {start_date} → {end_date}")
        prices = fetch_daily_prices(t, start_date, end_date)
        df = df.sort_values("published_utc").reset_index(drop=True)
        df = label_block(df, prices)
        blocks.append(df)

    if not blocks:
        raise SystemExit("No labeled data produced. Check your input files / tickers.")

    out = pd.concat(blocks, ignore_index=True)
    out = out.sort_values(["ticker", "published_utc"]).reset_index(drop=True)

    # Drop rows where 3d label is nan (no future prices)
    out = out.dropna(subset=["label_3d"])
    out.to_csv(out_csv, index=False)

    print(f"\nSaved labeled dataset → {out_csv} (rows: {len(out)})")
    print("Label distribution (3d):")
    print(out["label_3d"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
