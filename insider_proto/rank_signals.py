# rank_signals.py
#
# Module 3: Turn raw model predictions into human-usable signals.
#
# Input:  scored CSV from predict.py
#   required columns:
#     - ticker
#     - title
#     - publisher
#     - published_utc
#     - event_type
#     - prob_up_10d
#
# Also uses:
#   - event_type_stats_train.csv (from event_backtest.py)
#
# Output:
#   - ranked_signals.csv (or user-specified)
#     columns include:
#       ticker, title, event_type, prob_up_10d, signal_score, tier, reason
#
# Usage:
#   (.venv) python rank_signals.py --input scored_test.csv --output ranked_signals.csv
#   (.venv) python rank_signals.py --input scored_test.csv --top-frac 0.2

import argparse
import pandas as pd
import numpy as np


DEFAULT_EVENT_STATS = "event_type_stats_train.csv"


def load_scored(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"ticker", "title", "publisher", "published_utc", "event_type", "prob_up_10d"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input {path} is missing required columns: {missing}")

    # Ensure types are sane
    df["prob_up_10d"] = pd.to_numeric(df["prob_up_10d"], errors="coerce")
    df = df.dropna(subset=["prob_up_10d"]).reset_index(drop=True)

    return df


def load_event_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"event_type", "count", "avg_10d", "avg_20d", "hit_rate_10d"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Event stats file {path} is missing required columns: {missing}")

    return df


def compute_signal_scores(df_scored: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge scored headlines with event-type backtest stats
    and compute a composite signal_score.
    """

    # Merge on event_type
    df = df_scored.merge(
        stats[["event_type", "count", "avg_10d", "hit_rate_10d"]],
        on="event_type",
        how="left",
        suffixes=("", "_event"),
    )

    # Fallbacks: for event types not in stats (e.g., very rare)
    global_avg_10d = df["avg_10d"].mean(skipna=True)
    global_hit = df["hit_rate_10d"].mean(skipna=True)

    df["avg_10d"] = df["avg_10d"].fillna(global_avg_10d if not np.isnan(global_avg_10d) else 0.0)
    df["hit_rate_10d"] = df["hit_rate_10d"].fillna(global_hit if not np.isnan(global_hit) else 0.5)
    df["count"] = df["count"].fillna(0)

    # Base model probability
    df["base_prob"] = df["prob_up_10d"]

    # Event-edge components:
    # - avg_10d: expected sector-neutral 10d drift
    # - hit_rate_10d: how often this event_type actually goes up
    #
    # Normalize hit_rate around 0.5 so >0 is good, <0 is bad.
    df["win_edge"] = df["hit_rate_10d"] - 0.5

    # Scale avg_10d (which is in raw return space, like 0.01 = 1%)
    # so it's roughly comparable to probability adjustments.
    # Empirically: 10x avg_10d turns a +1% edge into +0.10 score bump.
    df["event_edge"] = df["avg_10d"] * 10.0

    # Confidence weight: larger sample size → more trust in event stats.
    # Simple saturating function: weight ~ 0 → 1 as count grows.
    df["conf_weight"] = (df["count"] / 500.0).clip(0.0, 1.0)

    # Composite score:
    # - start from base_prob
    # - add event_edge * conf_weight
    # - add smaller contribution from win_edge
    df["signal_score"] = (
        df["base_prob"]
        + df["event_edge"] * df["conf_weight"]
        + 0.2 * df["win_edge"] * df["conf_weight"]
    )

    # Reason string for UI / logs
    def build_reason(row):
        et = row.get("event_type", "unknown")
        avg10 = row.get("avg_10d", np.nan)
        hit = row.get("hit_rate_10d", np.nan)
        prob = row.get("prob_up_10d", np.nan)

        parts = [f"event={et}"]
        if not np.isnan(avg10):
            parts.append(f"hist_10d={avg10*100:.1f}%")
        if not np.isnan(hit):
            parts.append(f"hit={hit*100:.0f}%")
        if not np.isnan(prob):
            parts.append(f"model_prob={prob*100:.1f}%")
        return " | ".join(parts)

    df["reason"] = df.apply(build_reason, axis=1)

    return df


def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket signals into Tier A / B / C based on signal_score quantiles.
    """

    if len(df) == 0:
        df["tier"] = []
        return df

    q90 = df["signal_score"].quantile(0.9)
    q70 = df["signal_score"].quantile(0.7)

    def tier(score):
        if score >= q90:
            return "A"   # strongest
        elif score >= q70:
            return "B"
        else:
            return "C"

    df["tier"] = df["signal_score"].apply(tier)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Module 3: rank scored headlines into tradable signals."
    )
    parser.add_argument("--input", required=True, help="Input scored CSV from predict.py")
    parser.add_argument("--output", required=True, help="Output ranked signals CSV")
    parser.add_argument(
        "--event-stats",
        default=DEFAULT_EVENT_STATS,
        help=f"Event stats CSV (default: {DEFAULT_EVENT_STATS})",
    )
    parser.add_argument(
        "--top-frac",
        type=float,
        default=1.0,
        help="Fraction of rows to keep (0-1). Default 1.0 (keep all).",
    )

    args = parser.parse_args()

    print(f"Loading scored headlines from {args.input}")
    df_scored = load_scored(args.input)

    print(f"Loading event-type stats from {args.event_stats}")
    stats = load_event_stats(args.event_stats)

    print("Computing signal scores...")
    df_signals = compute_signal_scores(df_scored, stats)
    df_signals = assign_tiers(df_signals)

    # Sort by signal_score descending
    df_signals = df_signals.sort_values("signal_score", ascending=False).reset_index(drop=True)

    # Optionally keep only top fraction
    if 0 < args.top_frac < 1.0:
        k = max(1, int(len(df_signals) * args.top_frac))
        df_signals = df_signals.iloc[:k].copy()
        print(f"Keeping top {args.top_frac:.0%} → {k} rows.")

    print(f"Writing ranked signals → {args.output}")
    df_signals.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
