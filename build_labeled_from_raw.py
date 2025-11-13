# build_labeled_from_raw.py
# Sector-neutral 10d/20d labeling with robust date handling

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# ------------- Sector mapping -------------

SECTOR_MAP = {
    # Semis / chips
    "NVDA": "SMH", "AMD": "SMH", "MU": "SMH", "TSM": "SMH", "ASML": "SMH", "AVGO": "SMH",
    "TXN": "SMH", "QCOM": "SMH", "INTC": "SMH", "AMAT": "SMH", "LRCX": "SMH", "KLAC": "SMH",

    # General big tech
    "AAPL": "XLK", "MSFT": "XLK", "GOOG": "XLK", "GOOGL": "XLK", "META": "XLK",
    "ADBE": "XLK", "ORCL": "XLK", "CRM": "XLK", "SNOW": "XLK", "NET": "XLK",
    "DDOG": "XLK", "MDB": "XLK", "PATH": "XLK", "PLTR": "XLK", "NOW": "XLK",
    "TEAM": "XLK", "INTU": "XLK", "SQ": "XLK", "PYPL": "XLK",

    # Financials
    "JPM": "XLF", "GS": "XLF", "MS": "XLF", "C": "XLF", "BAC": "XLF", "WFC": "XLF",
    "AXP": "XLF", "SCHW": "XLF", "AMTD": "XLF", "COIN": "XLF", "BLK": "XLF", "IVZ": "XLF",

    # Consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "MCD": "XLY", "SBUX": "XLY",
    "DPZ": "XLY", "ETSY": "XLY", "EXPE": "XLY", "BKNG": "XLY", "TOL": "XLY",
    "ROKU": "XLY", "NKE": "XLY", "CMG": "XLY",

    # Consumer staples
    "PEP": "XLP", "KO": "XLP", "PG": "XLP", "COST": "XLP", "WMT": "XLP",
    "DLTR": "XLP", "DG": "XLP", "KHC": "XLP", "MDLZ": "XLP",

    # Industrials
    "BA": "XLI", "CAT": "XLI", "DE": "XLI", "GE": "XLI", "LMT": "XLI", "NOC": "XLI",
    "RTX": "XLI", "LHX": "XLI", "ETN": "XLI", "PH": "XLI", "HON": "XLI", "GD": "XLI",
    "TXT": "XLI", "PL": "XLI", "RKLB": "XLI",

    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "AEP": "XLU", "EXC": "XLU", "XEL": "XLU",
    "PEG": "XLU", "ED": "XLU",

    # Materials
    "LIN": "XLB", "FCX": "XLB", "NUE": "XLB", "DD": "XLB",

    # Health care
    "UNH": "XLV", "PFE": "XLV", "MRNA": "XLV", "REGN": "XLV", "VRTX": "XLV", "BMY": "XLV",
    "GILD": "XLV", "LLY": "XLV", "ABBV": "XLV",

    # Energy
    "XOM": "XLE", "CVX": "XLE", "SLB": "XLE", "HAL": "XLE",

    # Real estate
    "AMT": "XLRE", "PLD": "XLRE", "EQIX": "XLRE",
}

DEFAULT_SECTOR = "SPY"


# ------------- Helpers -------------

def safe_download(symbol: str, start: str, end: str):
    """Download daily close prices and return df indexed by date (python date)."""
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
    except Exception as e:
        print(f"ERROR downloading {symbol}: {e}")
        return None
    if df is None or df.empty:
        print(f"WARNING: no data for {symbol}")
        return None

    # Normalize index to python date
    df = df[["Close"]].copy()
    df.reset_index(inplace=True)
    # Column name could be 'Date' or something else depending on yfinance version
    if "Date" not in df.columns:
        # assume first column is the datetime
        date_col = df.columns[0]
    else:
        date_col = "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df[[date_col, "Close"]].dropna()
    df.set_index(date_col, inplace=True)
    return df


def price_at(df, d):
    """Return a SINGLE float price at or before date d."""
    if d in df.index:
        val = df.loc[d, "Close"]
    else:
        prev_idx = [x for x in df.index if x <= d]
        if not prev_idx:
            return None
        val = df.loc[prev_idx[-1], "Close"]

    # If val is a series (duplicates), take the last
    if isinstance(val, pd.Series):
        return float(val.iloc[-1])
    return float(val)


# ------------- Main -------------

def main():
    indir = "data/news_alpha"
    outfile = "news_labeled_alpha_sector_neutral.csv"

    files = [os.path.join(indir, f) for f in os.listdir(indir) if f.endswith(".csv")]
    if not files:
        raise SystemExit(f"No CSV files found in {indir}")

    today_str = datetime.today().strftime("%Y-%m-%d")

    # Pre-download sector ETF prices
    sector_universe = sorted(set(SECTOR_MAP.values()) | {DEFAULT_SECTOR})
    print("Downloading sector ETFs:", sector_universe)
    sector_prices = {}
    for etf in sector_universe:
        df = safe_download(etf, "2018-01-01", today_str)
        if df is None:
            continue
        sector_prices[etf] = df

    all_rows = []

    print("Processing", len(files), "news files...")
    for i, path in enumerate(files, 1):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR reading {path}: {e}")
            continue

        if df.empty:
            print(f"[{i}/{len(files)}] {path} empty, skip")
            continue

        if "ticker" not in df.columns or "published_utc" not in df.columns:
            print(f"[{i}/{len(files)}] {path} missing required columns, skip")
            continue

        ticker = str(df["ticker"].iloc[0]).upper()
        sector = SECTOR_MAP.get(ticker, DEFAULT_SECTOR)

        # price history for ticker
        p = safe_download(ticker, "2018-01-01", today_str)
        if p is None:
            print(f"[{i}/{len(files)}] {ticker}: no price data, skip")
            continue

        sec_df = sector_prices.get(sector)
        if sec_df is None:
            sec_df = sector_prices.get(DEFAULT_SECTOR)
            if sec_df is None:
                print(f"[{i}/{len(files)}] No sector/SPY data available, skip {ticker}")
                continue

        df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["published_utc"])
        if df.empty:
            print(f"[{i}/{len(files)}] {ticker}: no valid timestamps, skip")
            continue

        print(f"[{i}/{len(files)}] {ticker}: headlines={len(df)}")

        for _, row in df.iterrows():
            ts_dt = row["published_utc"]
            if pd.isna(ts_dt):
                continue
            ts = ts_dt.date()  # python date

            # find nearest trading date on or before ts
            p0 = price_at(p, ts)
            if p0 is None:
                continue

            d10 = ts + timedelta(days=10)
            d20 = ts + timedelta(days=20)

            p10 = price_at(p, d10)
            p20 = price_at(p, d20)
            if p10 is None or p20 is None:
                continue

            s0 = price_at(sec_df, ts)
            s10 = price_at(sec_df, d10)
            s20 = price_at(sec_df, d20)
            if s0 is None or s10 is None or s20 is None:
                continue

            ret10 = (p10 - p0) / p0
            ret20 = (p20 - p0) / p0
            sect10 = (s10 - s0) / s0
            sect20 = (s20 - s0) / s0

            alpha10 = ret10 - sect10
            alpha20 = ret20 - sect20

            label_10d_bin = int(float(alpha10) > 0)
            label_20d_bin = int(float(alpha20) > 0)


            all_rows.append({
                "ticker": ticker,
                "published_utc": row["published_utc"],
                "title": row.get("title", ""),
                "publisher": row.get("publisher", ""),
                "url": row.get("url", ""),
                "ret_10d": ret10,
                "ret_20d": ret20,
                "sect_10d": sect10,
                "sect_20d": sect20,
                "alpha_10d": alpha10,
                "alpha_20d": alpha20,
                "label_10d_bin": label_10d_bin,
                "label_20d_bin": label_20d_bin,
            })

    final = pd.DataFrame(all_rows)
    if final.empty:
        raise SystemExit("No labeled rows produced. Check data and mappings.")

    final.to_csv(outfile, index=False)
    print(f"Saved sector-neutral dataset → {outfile} (rows: {len(final)})")
    print("Label distribution (10d binary):")
    print(final["label_10d_bin"].value_counts())
    

if __name__ == "__main__":
    main()
