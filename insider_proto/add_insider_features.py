import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

IN_CSV = "news_labeled.csv"
OUT_CSV = "news_with_insiders.csv"

# === CONFIG ===
FINNHUB_KEY = os.getenv("FINNHUB_TOKEN") or "YOUR_API_KEY_HERE"  # <-- put your key here if not using env var
LOOKBACK_DAYS = 30

if FINNHUB_KEY == "YOUR_API_KEY_HERE":
    print("WARNING: FINNHUB_KEY not set. Edit add_insider_features.py and set FINNHUB_KEY.")

BASE_URL = "https://finnhub.io/api/v1/stock/insider-transactions"


def fetch_insiders(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch insider transactions for [start_date, end_date] from Finnhub.
    Returns DataFrame with columns: date, name, change, value, type
    """
    params = {
        "symbol": ticker,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "token": FINNHUB_KEY,
    }
    print(f"  Finnhub insiders {ticker} {params['from']} → {params['to']}")
    r = requests.get(BASE_URL, params=params, timeout=10)
    if r.status_code != 200:
        print(f"  WARNING: HTTP {r.status_code} for {ticker}: {r.text[:200]}")
        return pd.DataFrame(columns=["date", "name", "change", "value", "type"])

    data = r.json() or {}
    if "data" not in data or not data["data"]:
        print(f"  No insider data for {ticker}")
        return pd.DataFrame(columns=["date", "name", "change", "value", "type"])

    df_ins = pd.DataFrame(data["data"])

    # Normalize
    for col in ["name", "change", "transactionPrice", "transactionType", "transactionDate"]:
        if col not in df_ins.columns:
            df_ins[col] = np.nan

    df_ins["date"] = pd.to_datetime(df_ins["transactionDate"], errors="coerce").dt.date
    df_ins = df_ins.dropna(subset=["date"])
    df_ins["name"] = df_ins["name"].astype(str)
    df_ins["change"] = pd.to_numeric(df_ins["change"], errors="coerce")
    df_ins["transactionPrice"] = pd.to_numeric(df_ins["transactionPrice"], errors="coerce")

    df_ins["value"] = df_ins["change"] * df_ins["transactionPrice"]

    df_ins["type"] = df_ins["transactionType"].astype(str)

    df_ins = df_ins[["date", "name", "change", "value", "type"]].sort_values("date").reset_index(drop=True)
    return df_ins


def add_insider_features_for_ticker(block: pd.DataFrame, insiders: pd.DataFrame) -> pd.DataFrame:
    """
    For each headline, look back LOOKBACK_DAYS and compute insider stats.
    """
    if insiders.empty:
        for col in [
            "ins_net_shares_30d",
            "ins_net_value_30d",
            "ins_buy_trades_30d",
            "ins_sell_trades_30d",
            "ins_distinct_insiders_30d",
        ]:
            block[col] = np.nan
        return block

    dates = insiders["date"].values
    changes = insiders["change"].values
    values = insiders["value"].values
    types = insiders["type"].values
    names = insiders["name"].values

    net_shares = []
    net_value = []
    buys = []
    sells = []
    distinct_ins = []

    def window_indices(start_d, end_d):
        left = np.searchsorted(dates, start_d, side="left")
        right = np.searchsorted(dates, end_d, side="left")
        return left, right

    for ts in block["published_utc"]:
        d = ts.date()
        start_window = d - timedelta(days=LOOKBACK_DAYS)
        end_window = d  # exclude same-day trades

        l, r = window_indices(start_window, end_window)
        if l >= r:
            net_shares.append(0.0)
            net_value.append(0.0)
            buys.append(0)
            sells.append(0)
            distinct_ins.append(0)
            continue

        c_slice = changes[l:r]
        v_slice = values[l:r]
        t_slice = types[l:r]
        n_slice = names[l:r]

        is_buy = np.array([str(t).upper().startswith("P") for t in t_slice])
        is_sell = np.array([str(t).upper().startswith("S") for t in t_slice])

        buy_shares = c_slice[is_buy].sum() if is_buy.any() else 0.0
        sell_shares = -c_slice[is_sell].sum() if is_sell.any() else 0.0

        buy_val = v_slice[is_buy].sum() if is_buy.any() else 0.0
        sell_val = -v_slice[is_sell].sum() if is_sell.any() else 0.0

        net_shares.append(buy_shares - sell_shares)
        net_value.append(buy_val - sell_val)
        buys.append(int(is_buy.sum()))
        sells.append(int(is_sell.sum()))
        distinct_ins.append(len(set(n_slice)))

    block["ins_net_shares_30d"] = net_shares
    block["ins_net_value_30d"] = net_value
    block["ins_buy_trades_30d"] = buys
    block["ins_sell_trades_30d"] = sells
    block["ins_distinct_insiders_30d"] = distinct_ins
    return block


def main():
    print(f"Loading {IN_CSV}…")
    df = pd.read_csv(IN_CSV)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_utc", "ticker"]).reset_index(drop=True)
    df["ticker"] = df["ticker"].astype(str).str.upper()

    tickers = sorted(df["ticker"].unique())
    print(f"Headlines: {len(df)}, Tickers: {len(tickers)}")

    blocks = []
    for i, t in enumerate(tickers, 1):
        time.sleep(3)
        block = df[df["ticker"] == t].copy().reset_index(drop=True)
        if block.empty:
            continue

        min_ts = block["published_utc"].min()
        max_ts = block["published_utc"].max()

        start_date = (min_ts - timedelta(days=LOOKBACK_DAYS + 10))
        end_date = (max_ts + timedelta(days=1))

        print(f"\n[{i}/{len(tickers)}] {t}: headlines={len(block)}, insider range {start_date.date()} → {end_date.date()}")
        insiders = fetch_insiders(t, start_date, end_date)
        time.sleep(0.25)  # rate limit buffer

        block = add_insider_features_for_ticker(block, insiders)
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    out = out.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved with insider features → {OUT_CSV} (rows: {len(out)})")


if __name__ == "__main__":
    main()
