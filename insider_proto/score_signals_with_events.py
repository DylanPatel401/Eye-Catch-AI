# score_signals_with_events.py
#
# Take fresh scored signals (from predict.py with prob_up_10d + event_type)
# and combine:
#   - model probability
#   - event-type backtest stats (avg_10d, hit_rate_10d, count)
#   - publisher quality
# into a unified signal_score + tier (A/B/C).
#
# Usage:
#   (.venv) python score_signals_with_events.py \
#       --input fresh_scored.csv \
#       --output ranked_signals.csv \
#       --top 0.3

import argparse
import numpy as np
import pandas as pd


EVENT_STATS_CSV = "event_type_stats_train.csv"   # from event_backtest.py


# ---------- Publisher quality weighting ----------

def publisher_weight(publisher: str) -> float:
    """Heuristic quality score per publisher."""
    if not isinstance(publisher, str):
        return 1.0

    p = publisher.lower()

    tier1 = ["reuters", "bloomberg", "wall street journal", "wsj",
             "financial times", "ft.com", "associated press", "ap news"]
    tier2 = ["cnbc", "marketwatch", "yahoo finance", "investor's business daily",
             "ibd", "dow jones", "the motley fool"]

    if any(k in p for k in tier1):
        return 1.10
    if any(k in p for k in tier2):
        return 1.05
    if "seekingalpha" in p:
        return 0.98
    return 1.0


# ---------- Scoring logic ----------

def compute_signal_scores(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge event-type stats and compute:
      - base_prob (event hit rate)
      - win_edge (model_prob - base_prob)
      - event_edge (avg_10d)
      - conf_weight (log-scaled by count)
      - publisher_weight
      - model_component & event_component
      - signal_score_raw & normalized signal_score
      - tier (A/B/C)
    """

    # Build lookup from stats
    # stats columns expected: event_type, count, avg_10d, hit_rate_10d
    stats = stats.copy()
    stats["event_type"] = stats["event_type"].astype(str)

    # Global fallback base probability (weighted by count)
    total_count = stats["count"].sum()
    if total_count > 0:
        global_base_prob = (stats["hit_rate_10d"] * stats["count"]).sum() / total_count
    else:
        global_base_prob = 0.5

    # Merge by event_type
    df = df.copy()
    df["event_type"] = df["event_type"].astype(str)
    df = df.merge(
        stats[["event_type", "count", "avg_10d", "hit_rate_10d"]],
        on="event_type",
        how="left",
        suffixes=("", "_evt")
    )

    # Fill missing stats with "global baseline"
    df["count"] = df["count"].fillna(0)
    df["avg_10d"] = df["avg_10d"].fillna(0.0)
    df["hit_rate_10d"] = df["hit_rate_10d"].fillna(global_base_prob)



    # --- Handcrafted priors for sparse but obviously strong/weak events ---
    strong_priors = {
        "contract_major":  {"base_prob": 0.60, "avg_10d": 0.015, "min_count": 300},
        "regulatory_approval": {"base_prob": 0.55, "avg_10d": 0.012, "min_count": 200},
    }
    weak_priors = {
        "legal_risk": {"base_prob": 0.45, "avg_10d": 0.000, "min_count": 200},
    }

    for evt, cfg in strong_priors.items():
        mask = (df["event_type"] == evt) & (df["count"] < cfg["min_count"])
        df.loc[mask, "base_prob"] = cfg["base_prob"]
        df.loc[mask, "avg_10d"] = cfg["avg_10d"]
        df.loc[mask, "count"] = cfg["min_count"]

    for evt, cfg in weak_priors.items():
        mask = (df["event_type"] == evt) & (df["count"] < cfg["min_count"])
        df.loc[mask, "base_prob"] = cfg["base_prob"]
        df.loc[mask, "avg_10d"] = cfg["avg_10d"]
        df.loc[mask, "count"] = cfg["min_count"]




    # Base probability is event-type hit rate
    df["base_prob"] = df["hit_rate_10d"]

    # Win edge: how much higher the model is than historical hit rate
    df["win_edge"] = df["prob_up_10d"] - df["base_prob"]

    # Event edge: actual average 10d alpha from backtest
    df["event_edge"] = df["avg_10d"]

    # Confidence weight: small samples -> low weight, big buckets -> ~1
    # Using log scaling with cap at 1.0
    def _conf(c):
        c = max(float(c), 0.0)
        if c <= 0:
            return 0.0
        return min(1.0, np.log(c + 1.0) / np.log(5000.0))

    df["conf_weight"] = df["count"].apply(_conf)

    # Publisher quality
    df["pub_weight"] = df["publisher"].apply(publisher_weight)

    # Model component:
    #   - relative to 0.5 (neutral)
    #   - amplified or dampened by publisher quality
    df["model_component"] = (df["prob_up_10d"] - 0.5) * df["pub_weight"]

    # Event component:
    #   - win_edge scaled by confidence
    #   - plus the raw avg_10d (alpha) so that truly strong events get a bump
    df["event_component"] = df["win_edge"] * df["conf_weight"] + df["event_edge"]

    # Combine into a raw score
    # Heavier weight on model (0.6) vs event context (0.4)
    df["signal_score_raw"] = 0.6 * df["model_component"] + 0.4 * df["event_component"]

    # Normalize to 0–1 range for easier interpretation
    min_s = df["signal_score_raw"].min()
    max_s = df["signal_score_raw"].max()
    if max_s > min_s:
        df["signal_score"] = (df["signal_score_raw"] - min_s) / (max_s - min_s)
    else:
        df["signal_score"] = 0.5

    # Tiering: A / B / C based on normalized score
    def _tier(s):
        if s >= 0.75:
            return "A"
        if s >= 0.45:
            return "B"
        return "C"


    df["tier"] = df["signal_score"].apply(_tier)

    # Reason string for UI / debugging
    def _reason(row):
        return (
            f"event={row['event_type']} | "
            f"hist_10d={row['avg_10d']:.2%} | "
            f"hit={row['base_prob']:.0%} | "
            f"model_prob={row['prob_up_10d']:.1%} | "
            f"pub_w={row['pub_weight']:.2f}"
        )

    df["reason"] = df.apply(_reason, axis=1)

    return df


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Rank signals using event stats + publisher quality.")
    parser.add_argument("--input", required=True, help="Input CSV (from predict.py with prob_up_10d + event_type).")
    parser.add_argument("--output", required=True, help="Output CSV with scores + tiers.")
    parser.add_argument(
        "--top",
        type=float,
        default=1.0,
        help="Fraction of top rows to keep (0-1]. Default 1.0 = keep all.",
    )

    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input)

    if "prob_up_10d" not in df.columns:
        raise SystemExit("Input must contain 'prob_up_10d' column.")
    if "event_type" not in df.columns:
        raise SystemExit("Input must contain 'event_type' column (or add via classify_event before).")

    print(f"Loading event stats: {EVENT_STATS_CSV}")
    stats = pd.read_csv(EVENT_STATS_CSV)

    df_scored = compute_signal_scores(df, stats)

    # Sort by signal score desc
    df_scored = df_scored.sort_values("signal_score", ascending=False).reset_index(drop=True)

    # Optionally keep only top fraction
    if 0 < args.top < 1.0:
        k = max(1, int(len(df_scored) * args.top))
        df_scored = df_scored.iloc[:k].copy()

    df_scored.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
