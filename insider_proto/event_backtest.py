# event_backtest.py
#
# Backtest performance by event_type.
# Uses: news_with_insiders_sector_neutral.csv
# Requires: label_10d_bin, ret_10d, ret_20d, title, published_utc
#
# Output:
#   - Prints summary tables to console
#   - Writes event_type_stats_full.csv and event_type_stats_train.csv

import pandas as pd
import numpy as np

from event_types import classify_event

IN_CSV = "news_with_insiders_sector_neutral.csv"
MIN_COUNT = 50   # minimum samples per event_type to include in summary


def load_and_prepare():
    df = pd.read_csv(IN_CSV)

    required = {"label_10d_bin", "ret_10d", "ret_20d", "title", "published_utc"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {IN_CSV}: {missing}")

    df["label_10d_bin"] = df["label_10d_bin"].astype(float)
    df["ret_10d"] = df["ret_10d"].astype(float)
    df["ret_20d"] = df["ret_20d"].astype(float)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")

    df = df.dropna(
        subset=["published_utc", "title", "label_10d_bin", "ret_10d"]
    ).reset_index(drop=True)
    df["label_10d_bin"] = df["label_10d_bin"].astype(int)

    # Sort by time (same as training pipeline)
    df = df.sort_values("published_utc").reset_index(drop=True)

    # Event type classification (Module 1)
    df["event_type"] = df["title"].astype(str).apply(classify_event)

    return df


def summarize_by_event_type(df: pd.DataFrame, label: str, out_csv: str):
    # Group by event_type and compute stats
    g = (
        df.groupby("event_type")
        .agg(
            count=("label_10d_bin", "size"),
            avg_10d=("ret_10d", "mean"),
            med_10d=("ret_10d", "median"),
            avg_20d=("ret_20d", "mean"),
            hit_rate_10d=("label_10d_bin", lambda x: (x == 1).mean()),
        )
        .reset_index()
    )

    # Filter small sample sizes
    g = g[g["count"] >= MIN_COUNT].copy()

    # Sort by avg_10d descending
    g = g.sort_values("avg_10d", ascending=False).reset_index(drop=True)

    print(f"\n===== {label}: event_type stats (count >= {MIN_COUNT}) =====")
    print(g.head(20))

    g.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv} (rows: {len(g)})")


def main():
    print(f"Loading {IN_CSV}…")
    df = load_and_prepare()
    print(f"Total rows after cleaning: {len(df)}")
    print("Event type distribution (top 20):")
    print(df["event_type"].value_counts().head(20))

    # Time-based split: same 80/20 as train_xgb_model.py
    n = len(df)
    split_idx = int(n * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(f"\nTrain rows: {len(df_train)}  |  Test rows: {len(df_test)}")

    # Full sample stats
    summarize_by_event_type(df, "FULL SAMPLE", "event_type_stats_full.csv")

    # Train-only stats (safer for model calibration)
    summarize_by_event_type(df_train, "TRAIN ONLY", "event_type_stats_train.csv")


if __name__ == "__main__":
    main()
