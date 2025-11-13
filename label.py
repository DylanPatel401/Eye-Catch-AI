# label.py
# Label headlines with 1d/3d/7d forward returns using yfinance daily prices.

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

NEWS_CSV = "news_raw.csv"
OUT_CSV  = "news_labeled.csv"

# ========= 1. Load and clean news =========
news = pd.read_csv(NEWS_CSV)

required_cols = {"ticker", "published_utc", "title"}
missing = required_cols - set(news.columns)
if missing:
    raise SystemExit(f"news_raw.csv is missing required columns: {missing}")

news = news.dropna(subset=["ticker", "published_utc", "title"]).copy()
news["ticker"] = news["ticker"].astype(str).str.upper()
news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, errors="coerce")
news = news.dropna(subset=["published_utc"])

news = news.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
tickers = sorted(news["ticker"].unique())
print(f"Loaded {len(news)} headlines across {len(tickers)} tickers")

# ========= 2. Price fetch helper (robust) =========
def fetch_daily_prices(ticker: str, start_date, end_date):
    print(f"  yfinance download {ticker} {start_date} → {end_date}")
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        multi_level_index=False,
    )

    if df is None or df.empty:
        print(f"  WARNING: yfinance returned no data for {ticker}")
        return pd.DataFrame(columns=["date", "Close"])

    df = df.copy()
    # index is Date
    df["date"] = pd.to_datetime(df.index, errors="coerce").date

    if "Close" not in df.columns:
        print(f"  WARNING: no 'Close' column for {ticker}. cols={list(df.columns)}")
        return pd.DataFrame(columns=["date", "Close"])

    df = df.dropna(subset=["date"])
    if df.empty:
        print(f"  WARNING: all dates invalid for {ticker}")
        return pd.DataFrame(columns=["date", "Close"])

    out = (
        df[["date", "Close"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out

# ========= 3. Labeling helper =========
def label_block(news_block: pd.DataFrame, price_df: pd.DataFrame, band: float = 0.01):
    """
    Align each headline to next trading day close, compute forward returns and labels (+1/0/-1).
    Assumes price_df columns: date (python date), Close (float).
    """
    if price_df.empty:
        for col in ["ret_1d", "ret_3d", "ret_7d", "label_1d", "label_3d", "label_7d"]:
            news_block[col] = np.nan
        return news_block

    dates = price_df["date"].values
    closes = price_df["Close"].values

    ret_1d = []; ret_3d = []; ret_7d = []
    lab_1d = []; lab_3d = []; lab_7d = []

    def next_trading_index(ts):
        # Map headline timestamp → first trading day >= headline date
        d = ts.date()
        idx = np.searchsorted(dates, d)
        return idx if idx < len(dates) else None

    def label_return(r):
        if pd.isna(r):
            return np.nan
        if r > band:
            return 1
        if r < -band:
            return -1
        return 0

    for ts in news_block["published_utc"]:
        idx0 = next_trading_index(ts)
        if idx0 is None:
            r1 = r3 = r7 = l1 = l3 = l7 = np.nan
        else:
            p0 = closes[idx0]

            def fwd(h):
                j = idx0 + h
                return (closes[j] - p0) / p0 if j < len(closes) else np.nan

            r1 = fwd(1)
            r3 = fwd(3)
            r7 = fwd(7)

            l1 = label_return(r1)
            l3 = label_return(r3)
            l7 = label_return(r7)

        ret_1d.append(r1); ret_3d.append(r3); ret_7d.append(r7)
        lab_1d.append(l1); lab_3d.append(l3); lab_7d.append(l7)

    news_block["ret_1d"]   = ret_1d
    news_block["ret_3d"]   = ret_3d
    news_block["ret_7d"]   = ret_7d
    news_block["label_1d"] = lab_1d
    news_block["label_3d"] = lab_3d
    news_block["label_7d"] = lab_7d
    return news_block

# ========= 4. Run per ticker =========
labeled_parts = []

for i, t in enumerate(tickers, 1):
    block = news[news["ticker"] == t].copy().reset_index(drop=True)
    if block.empty:
        continue

    min_ts = block["published_utc"].min()
    max_ts = block["published_utc"].max()

    # pad a few days around range
    start_date = (min_ts - pd.Timedelta(days=5)).date()
    end_date   = (max_ts + pd.Timedelta(days=10)).date()

    print(f"[{i}/{len(tickers)}] {t} prices {start_date} → {end_date}")
    price_df = fetch_daily_prices(t, start_date, end_date)

    if price_df.empty:
        print(f"  WARNING: no usable price data for {t}, skipping.")
        continue

    labeled = label_block(block, price_df, band=0.01)
    labeled_parts.append(labeled)

if not labeled_parts:
    raise SystemExit("No labeled data produced; all tickers had empty price data. Check tickers and date ranges.")

out_df = pd.concat(labeled_parts, ignore_index=True)
out_df = out_df.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
out_df.to_csv(OUT_CSV, index=False)
print(f"Saved labeled dataset → {OUT_CSV} (rows: {len(out_df)})")
