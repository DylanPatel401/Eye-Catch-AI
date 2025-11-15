# score_signals_with_events.py
#
# Module 4: Full signal scoring engine.
#
# Takes:
#   - Input: CSV from predict.py with at least:
#       ticker, title, publisher, published_utc, event_type, prob_up_10d
#   - Event stats: event_type_stats_train.csv from event_backtest.py
#
# Produces:
#   - Same rows +:
#       base_prob        (historical hit-rate for this event type)
#       win_edge         (model_prob - base_prob)
#       event_edge       (avg sector-neutral 10d alpha for this event)
#       conf_weight      (how statistically reliable this event type is)
#       signal_score     (final composite signal)
#       reason           (human-readable explanation)
#       tier             (A/B/C/D)
#
# Usage:
#   (.venv) python score_signals_with_events.py \
#       --input fresh_scored.csv \
#       --output ranked_signals.csv

import argparse
import numpy as np
import pandas as pd


EVENT_STATS_CSV = "event_type_stats_train.csv"


def load_event_stats(path: str) -> pd.DataFrame:
    """Load per-event stats and compute global baseline."""
    stats = pd.read_csv(path)

    required = {"event_type", "count", "avg_10d", "hit_rate_10d"}
    missing = required - set(stats.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path}: {missing}")

    # Clean
    stats["count"] = stats["count"].fillna(0).astype(float)
    stats["avg_10d"] = stats["avg_10d"].fillna(0.0).astype(float)
    stats["hit_rate_10d"] = stats["hit_rate_10d"].fillna(0.0).astype(float)

    # Global baseline = count-weighted hit-rate
    total_count = stats["count"].sum()
    if total_count <= 0:
        global_hit = stats["hit_rate_10d"].mean()
    else:
        global_hit = (stats["hit_rate_10d"] * stats["count"]).sum() / total_count

    return stats, float(global_hit)


def compute_signal_scores(df_in: pd.DataFrame,
                          stats: pd.DataFrame,
                          global_hit: float) -> pd.DataFrame:
    """
    Join model outputs with historical event stats and compute:
      base_prob, win_edge, event_edge, conf_weight, signal_score, reason, tier.
    """

    if "event_type" not in df_in.columns:
        raise SystemExit("Input must contain 'event_type' column. Run predict.py (with event_type) first.")
    if "prob_up_10d" not in df_in.columns:
        raise SystemExit("Input must contain 'prob_up_10d' column from model scoring.")

    # Merge on event_type
    stats_small = stats[["event_type", "count", "avg_10d", "hit_rate_10d"]].copy()
    df = df_in.merge(stats_small, on="event_type", how="left")

    # Fill missing stats with global baseline
    df["count"] = df["count"].fillna(0.0)
    df["avg_10d"] = df["avg_10d"].fillna(0.0)
    df["hit_rate_10d"] = df["hit_rate_10d"].fillna(global_hit)

    # Base probability = historical hit-rate for this event type
    df["base_prob"] = df["hit_rate_10d"].astype(float)

    # Edges
    df["prob_up_10d"] = df["prob_up_10d"].astype(float)
    df["win_edge"] = df["prob_up_10d"] - df["base_prob"]          # model vs. historical
    df["event_edge"] = df["avg_10d"].astype(float)                # avg sector-neutral 10d alpha

    # Confidence weight based on sample size of this event type
    # log10(count+1)/2 clipped to [0.1, 1.0]
    conf = np.log10(df["count"] + 1.0) / 2.0
    df["conf_weight"] = conf.clip(lower=0.1, upper=1.0)

    # Composite raw score:
    # - 55% model probability
    # - 30% uplift vs. baseline (win_edge)
    # - 15% scaled event-level alpha
    df["signal_score_raw"] = (
        0.55 * df["prob_up_10d"]
        + 0.30 * df["win_edge"]
        + 0.15 * (df["event_edge"] * 10.0)   # 1% alpha ≈ 0.1 contribution
    )

    # Final score = raw_score * confidence
    df["signal_score"] = df["signal_score_raw"] * df["conf_weight"]

    # Reason string
    def make_reason(row):
        return (
            f"event={row['event_type']} | "
            f"hist_10d={row['avg_10d']:.2%} | "
            f"hit={row['hit_rate_10d']:.0%} | "
            f"model_prob={row['prob_up_10d']:.1%}"
        )

    df["reason"] = df.apply(make_reason, axis=1)

    # Tiering: percentile-based on signal_score + hard guards on prob/edge/conf
    scores = df["signal_score"].fillna(df["signal_score"].min())
    q90, q75, q50 = np.quantile(scores, [0.90, 0.75, 0.50])

    def assign_tier(row):
        s = row["signal_score"]
        p = row["prob_up_10d"]
        w = row["win_edge"]
        c = row["conf_weight"]

        if s >= q90 and p >= 0.60 and w >= 0.05 and c >= 0.5:
            return "A"
        elif s >= q75 and p >= 0.55 and w >= 0.02:
            return "B"
        elif s >= q50:
            return "C"
        else:
            return "D"

    df["tier"] = df.apply(assign_tier, axis=1)

    # Sort strongest first
    df = df.sort_values("signal_score", ascending=False).reset_index(drop=True)

    return df


def main():
    ap = argparse.ArgumentParser(description="Rank signals with event-aware scoring engine.")
    ap.add_argument("--input", required=True,
                    help="Input CSV from predict.py (must include event_type, prob_up_10d).")
    ap.add_argument("--output", required=True,
                    help="Output CSV with signal_score, tier, etc.")
    ap.add_argument("--event-stats", default=EVENT_STATS_CSV,
                    help=f"Event stats CSV (default: {EVENT_STATS_CSV}).")
    ap.add_argument("--top", type=float, default=1.0,
                    help="Fraction of rows to keep (0-1]. Default 1.0 (keep all).")

    args = ap.parse_args()

    print(f"Loading event stats from {args.event_stats} …")
    stats, global_hit = load_event_stats(args.event_stats)
    print(f"Global baseline hit-rate (10d up): {global_hit:.3%}")

    print(f"Reading input: {args.input}")
    df_in = pd.read_csv(args.input)

    scored = compute_signal_scores(df_in, stats, global_hit)

    # Optional filter to top fraction
    if 0 < args.top < 1.0:
        k = max(1, int(len(scored) * args.top))
        scored = scored.iloc[:k].copy()
        print(f"Keeping top {args.top:.0%} → {k} rows.")

    print(f"Writing ranked signals → {args.output}")
    scored.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
