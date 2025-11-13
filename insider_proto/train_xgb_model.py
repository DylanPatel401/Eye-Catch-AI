# train_xgb_model.py
#
# XGBoost model on 10-day binary label:
# - Input: news_with_insiders.csv (from add_insider_features.py)
# - Target: label_10d_bin (1 if ret_10d > +2%, else 0)
# - Features: TF-IDF(title) + keyword flags + publisher/time + insider features
# - Split: time-based 80/20

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

FRICTION = 0.0035   # 0.35% costs
TOP_K = 0.20        # top decile

IN_CSV = "news_with_insiders.csv"

# ==================== Load =====================

df = pd.read_csv(IN_CSV)

required = {"label_10d_bin", "ret_10d", "ret_20d", "title", "published_utc"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {IN_CSV}: {missing}")

df["label_10d_bin"] = df["label_10d_bin"].astype(float)
df["ret_10d"] = df["ret_10d"].astype(float)
df["ret_20d"] = df["ret_20d"].astype(float)
df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")

df = df.dropna(subset=["published_utc", "title", "label_10d_bin", "ret_10d"]).reset_index(drop=True)
df["label_10d_bin"] = df["label_10d_bin"].astype(int)

print("Total labeled rows:", len(df))
print("Label distribution (10d binary):")
print(df["label_10d_bin"].value_counts(dropna=False))

# Sort by time for time-based split
df = df.sort_values("published_utc").reset_index(drop=True)

# ==================== Feature engineering =====================

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
    t = text.lower()
    return any(w in t for w in words)


# keyword flags
for key, words in KEYWORDS.items():
    df[f"kw_{key}"] = df["title"].astype(str).apply(
        lambda x: 1 if contains_any(x, words) else 0
    )

# publisher flags
pub = df["publisher"].fillna("").astype(str)
df["is_sa"] = pub.str.contains("SeekingAlpha", case=False).astype(int)
df["is_yahoo"] = pub.str.contains("Yahoo", case=False).astype(int)

# time features
df["hour"] = df["published_utc"].dt.hour
df["dow"] = df["published_utc"].dt.dayofweek

# insider features (from add_insider_features.py)
insider_cols = [
    "ins_net_shares_30d",
    "ins_net_value_30d",
    "ins_buy_trades_30d",
    "ins_sell_trades_30d",
    "ins_distinct_insiders_30d",
]
for c in insider_cols:
    if c not in df.columns:
        raise SystemExit(f"Missing insider feature {c} in {IN_CSV}")

y = df["label_10d_bin"]

structured_cols = (
    [c for c in df.columns if c.startswith("kw_")] +
    ["is_sa", "is_yahoo", "hour", "dow"] +
    insider_cols
)

print("Structured feature columns:", structured_cols)

# ==================== Text features =====================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
)

X_text = tfidf.fit_transform(df["title"].astype(str))
X_struct = df[structured_cols].fillna(0).values

X = hstack([X_text, X_struct])
X = csr_matrix(X)

# ==================== Time-based split =====================

n_samples = len(df)
split_idx = int(n_samples * 0.8)

X_train = X[:split_idx]
X_test = X[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ==================== Train XGBoost =====================

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
if pos == 0:
    raise SystemExit("No positive samples in training set.")
scale_pos_weight = neg / pos if pos > 0 else 1.0
print(f"scale_pos_weight: {scale_pos_weight:.3f} (neg={neg}, pos={pos})")

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
)

model.fit(X_train, y_train)

# ==================== Evaluation =====================

proba = model.predict_proba(X_test)[:, 1]
df_test["prob_up_10d"] = proba

n_test = len(df_test)
k = max(1, int(n_test * TOP_K))

top = df_test.nlargest(k, "prob_up_10d").copy()

# Only evaluate where we have returns
top_valid = top.dropna(subset=["ret_10d", "ret_20d"])
avg_10d_gross = top_valid["ret_10d"].mean()
avg_10d_net = avg_10d_gross - FRICTION
avg_20d_gross = top_valid["ret_20d"].mean()
avg_20d_net = avg_20d_gross - FRICTION
hit_rate_10d = (top_valid["label_10d_bin"] == 1).mean()

print(f"\nTop {int(TOP_K*100)}% (n={len(top_valid)}) stats (time-based split):")
print(f"  Avg 10d gross return: {avg_10d_gross:.3%}")
print(f"  Avg 10d net return (after {FRICTION:.2%} friction): {avg_10d_net:.3%}")
print(f"  Avg 20d gross return: {avg_20d_gross:.3%}")
print(f"  Avg 20d net return (after {FRICTION:.2%} friction): {avg_20d_net:.3%}")
print(f"  Hit rate (10d up): {hit_rate_10d:.3f}")

print("\nTop-bucket ticker concentration:")
print(top_valid["ticker"].value_counts().head(15))

# Classification metrics
pred_labels = (proba >= 0.5).astype(int)
print("\nClassification report (10d binary):")
print(classification_report(y_test, pred_labels))
