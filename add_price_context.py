# add_price_context.py
# Add pre-news price & volume context features to news_labeled.csv

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

IN_CSV = "news_labeled.csv"
OUT_CSV = "news_with_context.csv"

print("Loading labeled news...")
df = pd.read_csv(IN_CSV)

required = {"ticker", "published_utc", "ret_3d", "label_3d"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {IN_CSV}: {missing}")

df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["published_utc", "ticker"]).reset_index(drop=True)
df["ticker"] = df["ticker"].astype(str).str.upper()

tickers = sorted(df["ticker"].unique())
print(f"Headlines: {len(df)}, Tickers: {len(tickers)}")

# ---------------- PRICE FETCH (ROBUST) ---------------- #

def fetch_daily_prices(ticker: str, start_date, end_date):
    """
    Fetch daily adjusted Close & Volume for [start_date, end_date].
    Handles both flat and MultiIndex columns from yfinance.
    Returns DataFrame: date, Close, Volume
    """
    print(f"  yfinance download {ticker} {start_date} → {end_date}")
    prices = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),  # end is exclusive
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if prices is None or prices.empty:
        print(f"  WARNING: no data for {ticker}")
        return pd.DataFrame(columns=["date", "Close", "Volume"])

    prices = prices.copy()

    # --- Case 1: MultiIndex columns, like ('Close','ACLS'), ('Volume','ACLS') ---
    if isinstance(prices.columns, pd.MultiIndex):
        level0 = prices.columns.get_level_values(0)

        if "Close" in level0:
            close_df = prices.xs("Close", axis=1, level=0)
        else:
            close_df = None
        if "Volume" in level0:
            vol_df = prices.xs("Volume", axis=1, level=0)
        else:
            vol_df = None

        if close_df is None or vol_df is None or close_df.empty or vol_df.empty:
            print(f"  WARNING: missing Close/Volume in MultiIndex for {ticker} (cols={list(prices.columns)})")
            return pd.DataFrame(columns=["date", "Close", "Volume"])

        # For a single ticker, each is usually a 1-col DataFrame; take first column.
        close_series = close_df.iloc[:, 0]
        vol_series = vol_df.iloc[:, 0]

    # --- Case 2: Flat columns, e.g. 'Close', 'Volume' ---
    else:
        cols = list(prices.columns)
        if "Close" not in cols or "Volume" not in cols:
            print(f"  WARNING: missing Close/Volume for {ticker} (cols={cols})")
            return pd.DataFrame(columns=["date", "Close", "Volume"])
        close_series = prices["Close"]
        vol_series = prices["Volume"]

    # Build date from index
    idx = prices.index
    date_values = pd.to_datetime(idx, errors="coerce").date

    out = pd.DataFrame({
        "date": date_values,
        "Close": close_series.values,
        "Volume": vol_series.values,
    })

    out = (
        out.dropna(subset=["date"])
           .drop_duplicates(subset=["date"])
           .sort_values("date")
           .reset_index(drop=True)
    )

    if out.empty:
        print(f"  WARNING: empty after cleaning for {ticker}")
    return out

# ---------------- CONTEXT FEATURES ---------------- #

def add_context_for_ticker(block: pd.DataFrame, prices: pd.DataFrame):
    """
    For each headline, compute:
      ret_-1d, mom_5d, mom_20d, rvol_5d, rvol_20d
    using prices BEFORE the headline date.
    """
    if prices.empty:
        for col in ["ret_-1d", "mom_5d", "mom_20d", "rvol_5d", "rvol_20d"]:
            block[col] = np.nan
        return block

    dates = prices["date"].values
    close = prices["Close"].values
    vol   = prices["Volume"].values

    ret_1d = []
    mom_5d = []
    mom_20d = []
    rvol_5d = []
    rvol_20d = []

    def last_trading_index_before(ts):
        d = ts.date()
        # index of first trading day > d
        idx = np.searchsorted(dates, d, side="right")
        j = idx - 1
        if j < 0:
            return None
        return j

    for ts in block["published_utc"]:
        j0 = last_trading_index_before(ts)
        if j0 is None:
            ret_1d.append(np.nan)
            mom_5d.append(np.nan)
            mom_20d.append(np.nan)
            rvol_5d.append(np.nan)
            rvol_20d.append(np.nan)
            continue

        p0 = close[j0]
        v0 = vol[j0]

        # 1-day return up to p0
        if j0 - 1 >= 0:
            p_prev = close[j0 - 1]
            ret_1d.append((p0 - p_prev) / p_prev if p_prev != 0 else np.nan)
        else:
            ret_1d.append(np.nan)

        # 5-day momentum
        if j0 - 5 >= 0:
            p_5 = close[j0 - 5]
            mom_5d.append((p0 - p_5) / p_5 if p_5 != 0 else np.nan)
        else:
            mom_5d.append(np.nan)

        # 20-day momentum
        if j0 - 20 >= 0:
            p_20 = close[j0 - 20]
            mom_20d.append((p0 - p_20) / p_20 if p_20 != 0 else np.nan)
        else:
            mom_20d.append(np.nan)

        # 5-day RVOL
        if j0 - 5 >= 0:
            v_window = vol[j0 - 5:j0]
            m = v_window.mean()
            rvol_5d.append(v0 / m if m not in (0, np.nan) else np.nan)
        else:
            rvol_5d.append(np.nan)

        # 20-day RVOL
        if j0 - 20 >= 0:
            v_window = vol[j0 - 20:j0]
            m = v_window.mean()
            rvol_20d.append(v0 / m if m not in (0, np.nan) else np.nan)
        else:
            rvol_20d.append(np.nan)

    block["ret_-1d"]  = ret_1d
    block["mom_5d"]   = mom_5d
    block["mom_20d"]  = mom_20d
    block["rvol_5d"]  = rvol_5d
    block["rvol_20d"] = rvol_20d
    return block

# ---------------- MAIN LOOP ---------------- #

blocks = []
for i, t in enumerate(tickers, 1):
    t_block = df[df["ticker"] == t].copy().reset_index(drop=True)
    if t_block.empty:
        continue

    min_ts = t_block["published_utc"].min()
    max_ts = t_block["published_utc"].max()

    # get enough history before earliest headline
    start_date = (min_ts - pd.Timedelta(days=40)).date()
    end_date   = (max_ts + pd.Timedelta(days=1)).date()

    print(f"\n[{i}/{len(tickers)}] {t}: headlines={len(t_block)}, price range {start_date} → {end_date}")
    prices = fetch_daily_prices(t, start_date, end_date)
    t_block = add_context_for_ticker(t_block, prices)
    blocks.append(t_block)

out = pd.concat(blocks, ignore_index=True)
out = out.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved with context → {OUT_CSV} (rows: {len(out)})")
