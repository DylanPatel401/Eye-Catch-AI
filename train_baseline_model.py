# train_baseline_model.py
#
# Baseline model with:
# - Time-based split (oldest 80% train, newest 20% test)
# - TF-IDF text features
# - Keyword/publisher/time features
# - Pre-news price/volume context: ret_-1d, mom_5d, mom_20d, rvol_5d, rvol_20d

import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

FRICTION = 0.0035
TOP_K = 0.10

IN_CSV = "news_with_context.csv"

# ==================== Load & clean =====================
df = pd.read_csv(IN_CSV)

required = {"label_3d", "ret_3d", "title", "published_utc"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {IN_CSV}: {missing}")

df["label_3d"] = df["label_3d"].astype(float)
df["ret_3d"] = df["ret_3d"].astype(float)
df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce")

df = df.dropna(subset=["published_utc", "title", "label_3d", "ret_3d"]).reset_index(drop=True)
df["label_3d"] = df["label_3d"].astype(int)

print("Total labeled rows:", len(df))
print("Label distribution (3d):")
print(df["label_3d"].value_counts(dropna=False))

# Sort by time
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
    "recall": ["recall", "defect"],
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

# context features (already added by add_price_context.py)
context_cols = ["ret_-1d", "mom_5d", "mom_20d", "rvol_5d", "rvol_20d"]
for c in context_cols:
    if c not in df.columns:
        raise SystemExit(f"Missing context feature {c} in {IN_CSV}")

# target
y = df["label_3d"]

# structured feature columns
structured_cols = (
    [c for c in df.columns if c.startswith("kw_")] +
    ["is_sa", "is_yahoo", "hour", "dow"] +
    context_cols
)

print("Structured feature columns:", structured_cols)

# ==================== Text features =====================

tfidf = TfidfVectorizer(
    max_features=3000,
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

# ==================== Train model =====================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ==================== Evaluation =====================

proba = model.predict_proba(X_test)
classes = list(model.classes_)
if 1 in classes:
    idx_up = classes.index(1)
else:
    raise RuntimeError(f"Model classes {classes} do not contain label 1")

df_test["prob_up"] = proba[:, idx_up]

n_test = len(df_test)
k = max(1, int(n_test * TOP_K))
top = df_test.nlargest(k, "prob_up")

avg_3d_gross = top["ret_3d"].mean()
avg_3d_net = avg_3d_gross - FRICTION
hit_rate = (top["label_3d"] == 1).mean()

print(f"\nTop {int(TOP_K*100)}% (n={k}) stats (time-based split):")
print(f"  Avg 3d gross return: {avg_3d_gross:.3%}")
print(f"  Avg 3d net return (after {FRICTION:.2%} friction): {avg_3d_net:.3%}")
print(f"  Hit rate (label_3d == 1): {hit_rate:.3f}")

print("\nTop-bucket ticker concentration:")
print(top["ticker"].value_counts().head(10))

preds = model.predict(X_test)
print("\nClassification report (3-class -1/0/1 on 3d move):")
print(classification_report(y_test, preds))
