# predict.py
#
# Use trained XGBoost sector-neutral model to score NEW headlines.
# Input:  CSV with at least: ticker, title, publisher, published_utc
# Output: same rows + event_type + prob_up_10d, sorted by probability desc.
#
# Usage (from insider_proto):
#   (.venv) python predict.py --input fresh_news.csv --output scored_signals.csv

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
import joblib

from event_types import classify_event


MODEL_PATH = "xgb_sector_neutral_v01.pkl"
VECTORIZER_PATH = "tfidf_sector_neutral_v01.pkl"
STRUCT_COLS_PATH = "struct_cols_v01.pkl"


# --- keyword setup (must match train_xgb_model.py) ---

KEYWORDS = {
    "partnership": ["partner", "partnership", "collaboration"],
    "invests": ["invest", "funding", "stake"],
    "acquisition": ["acquire", "acquisition", "merger", "buyout"],
    "contract": ["contract", "award"],
    "guidance": ["guidance", "outlook"],
    "downgrade": ["downgrade", "cuts rating"],
    "lawsuit": ["lawsuit", "sues", "settlement"],
    "recall": ["recall", "defect", "safety issue"],
}


def contains_any(text: str, words) -> bool:
    t = str(text).lower()
    return any(w in t for w in words)


def build_features(df: pd.DataFrame, struct_cols) -> tuple[pd.DataFrame, csr_matrix]:
    """
    Build structured features to match struct_cols from training.
    - Adds kw_* flags
    - Adds is_sa, is_yahoo
    - Adds hour, dow
    - Leaves insider cols as-is if present, else fills 0
    """
    # Ensure required base columns exist
    if "title" not in df.columns:
        raise SystemExit("Input CSV must have a 'title' column.")
    if "publisher" not in df.columns:
        df["publisher"] = ""

    if "published_utc" not in df.columns:
        raise SystemExit("Input CSV must have a 'published_utc' column (ISO string).")

    # Parse timestamp
    df["published_utc"] = pd.to_datetime(
        df["published_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=["published_utc"]).reset_index(drop=True)

    # keyword flags (recompute)
    for key, words in KEYWORDS.items():
        col = f"kw_{key}"
        df[col] = df["title"].astype(str).apply(
            lambda x: 1 if contains_any(x, words) else 0
        )

    # publisher flags
    pub = df["publisher"].fillna("").astype(str)
    df["is_sa"] = pub.str.contains("SeekingAlpha", case=False).astype(int)
    df["is_yahoo"] = pub.str.contains("Yahoo", case=False).astype(int)

    # time features
    df["hour"] = df["published_utc"].dt.hour
    df["dow"] = df["published_utc"].dt.dayofweek

    # Ensure all struct_cols exist; missing ones become 0
    for c in struct_cols:
        if c not in df.columns:
            df[c] = 0

    X_struct = csr_matrix(df[struct_cols].fillna(0).values)
    return df, X_struct


def main():
    parser = argparse.ArgumentParser(
        description="Score fresh news with XGB sector-neutral model."
    )
    parser.add_argument("--input", required=True, help="Input CSV with fresh news.")
    parser.add_argument("--output", required=True, help="Output CSV with scores.")
    parser.add_argument(
        "--top",
        type=float,
        default=1.0,
        help="Fraction of rows to keep (0-1). Default: 1.0 (keep all).",
    )

    args = parser.parse_args()

    print(f"Loading model artifacts: {MODEL_PATH}, {VECTORIZER_PATH}, {STRUCT_COLS_PATH}")
    model = joblib.load(MODEL_PATH)
    vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
    struct_cols = joblib.load(STRUCT_COLS_PATH)

    print(f"Reading input: {args.input}")
    df_raw = pd.read_csv(args.input)

    if "ticker" not in df_raw.columns:
        raise SystemExit("Input CSV must contain a 'ticker' column.")

    if "title" not in df_raw.columns:
        raise SystemExit("Input CSV must contain a 'title' column.")

    # Module 1: event type classification
    df_raw["event_type"] = df_raw["title"].astype(str).apply(classify_event)

    # Build structured features
    df_feat, X_struct = build_features(df_raw.copy(), struct_cols)

    # Text features (use trained vectorizer, DO NOT fit again)
    X_text = vectorizer.transform(df_feat["title"].astype(str))

    # Combine
    X = hstack([X_text, X_struct]).tocsr()

    # Predict
    print("Scoring...")
    proba = model.predict_proba(X)[:, 1]
    df_feat["prob_up_10d"] = proba

    # Merge scores back to original rows (index align)
    df_out = df_raw.loc[df_feat.index].copy()
    df_out["prob_up_10d"] = df_feat["prob_up_10d"].values

    # Sort by probability
    df_out = df_out.sort_values("prob_up_10d", ascending=False).reset_index(drop=True)

    # Optionally keep only top fraction
    if 0 < args.top < 1.0:
        k = max(1, int(len(df_out) * args.top))
        df_out = df_out.iloc[:k].copy()
        print(f"Keeping top {args.top:.0%} → {k} rows.")

    print(f"Writing scored output -> {args.output}")
    df_out.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
