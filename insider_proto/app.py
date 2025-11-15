# app.py
#
# Streamlit frontend for your Eye-Catch-AI prototype.
#
# What it does:
# - Option 1: Paste URLs → auto-fetch title, publisher, published_utc, best-guess ticker.
# - Option 2: Manually edit a table (ticker, title, publisher, published_utc).
# - Writes them to a temporary CSV.
# - Calls your existing CLI pipeline:
#       python predict.py --input tmp_in.csv --output tmp_scored.csv
#       python score_signals_with_events.py --input tmp_scored.csv --output tmp_ranked.csv --top 1.0
# - Loads ranked signals and shows them in a sortable table.
# - Lets you download the ranked CSV.
#
# Requirements:
#   - This file must live in the SAME FOLDER as:
#       predict.py
#       score_signals_with_events.py
#       event_types.py
#       xgb_sector_neutral_v01.pkl
#       tfidf_sector_neutral_v01.pkl
#       struct_cols_v01.pkl
#       url_ingest.py  (with urls_to_df defined)
#
# Run with:
#   (.venv) streamlit run app.py

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from url_ingest import urls_to_df  # <-- URL helper

BASE_DIR = Path(__file__).resolve().parent
PREDICT_SCRIPT = BASE_DIR / "predict.py"
SCORE_SCRIPT = BASE_DIR / "score_signals_with_events.py"


def run_subprocess(cmd, cwd=None):
    """
    Run a subprocess command and return (ok, stdout, stderr).
    Uses sys.executable so it runs inside your active venv.
    """
    try:
        result = subprocess.run(
            [sys.executable] + cmd,
            cwd=cwd or BASE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        ok = result.returncode == 0
        return ok, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def make_default_rows():
    """Starter rows so you don't see an empty table."""
    now_iso = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "title": "Nvidia announces new AI superchip contract with US government",
                "publisher": "Reuters",
                "published_utc": now_iso,
            },
            {
                "ticker": "PLTR",
                "title": "Palantir wins billion-dollar analytics contract with US government agency",
                "publisher": "Yahoo Finance",
                "published_utc": now_iso,
            },
            {
                "ticker": "AAPL",
                "title": "Apple warns of iPhone supply constraints for next quarter",
                "publisher": "Bloomberg",
                "published_utc": now_iso,
            },
        ]
    )


def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean the input table:
    - Strip whitespace
    - Uppercase tickers
    - Drop rows with missing ticker or title
    - Ensure published_utc is present (if blank, fill with "now")
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ["ticker", "title", "publisher", "published_utc"]:
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["title"] = df["title"].astype(str).str.strip()
    df["publisher"] = df["publisher"].astype(str).str.strip()

    # Fill missing times with now
    now_iso = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    df["published_utc"] = df["published_utc"].astype(str).str.strip()
    df.loc[df["published_utc"] == "", "published_utc"] = now_iso

    # Drop empty rows
    df = df[(df["ticker"] != "") & (df["title"] != "")]
    df = df.reset_index(drop=True)
    return df[["ticker", "title", "publisher", "published_utc"]]


def main():
    st.set_page_config(
        page_title="Eye-Catch-AI: Signal Tester",
        layout="wide",
    )

    st.title("📈 Eye-Catch-AI — Signal Tester")
    st.caption(
        "Paste URLs or edit headlines → run your XGBoost + event backtest pipeline → get ranked signals."
    )

    st.markdown(
        """
        **Two ways to feed headlines into the model:**

        1. **From URLs tab** – paste multiple article URLs (Yahoo, Bloomberg, etc.).  
           I’ll fetch: title, publisher, publish time, and a best-guess ticker. You can edit everything before scoring.

        2. **Manual table tab** – directly edit the table of `ticker, title, publisher, published_utc`.

        Then click **“Run model & rank signals”** to run:
        `predict.py` → `score_signals_with_events.py`.
        """
    )

    with st.sidebar:
        st.header("Options")
        top_fraction = st.slider(
            "Top fraction to keep (score_signals_with_events.py --top)",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="This is passed as --top to score_signals_with_events.py. 1.0 = keep all.",
        )

        st.markdown("---")
        st.markdown(
            """
            **Files expected in this folder:**
            - `predict.py`
            - `score_signals_with_events.py`
            - `event_types.py`
            - `xgb_sector_neutral_v01.pkl`
            - `tfidf_sector_neutral_v01.pkl`
            - `struct_cols_v01.pkl`
            - `url_ingest.py`
            """
        )

    # --- Initialize session state ---
    if "input_df" not in st.session_state:
        st.session_state["input_df"] = make_default_rows()

    # --- Tabs: From URLs vs Manual input ---
    tab_urls, tab_manual = st.tabs(["🔗 From URLs", "✍️ Manual table"])

    with tab_urls:
        st.subheader("Paste article URLs")
        st.markdown(
            "Paste **one URL per line** (Yahoo Finance, Bloomberg, Reuters, etc.). "
            "I’ll try to infer `title`, `publisher`, `published_utc`, and a ticker guess."
        )

        urls_text = st.text_area(
            "URLs",
            height=180,
            placeholder="https://finance.yahoo.com/news/... \nhttps://www.bloomberg.com/news/articles/...",
        )

        if st.button("Fetch headlines from URLs"):
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not urls:
                st.warning("No URLs provided.")
            else:
                with st.spinner("Fetching metadata from URLs…"):
                    df_urls = urls_to_df(urls)

                if df_urls.empty:
                    st.error("No articles could be loaded from those URLs.")
                else:
                    # Only keep core columns for the model; keep extras like source_url if present
                    base_cols = ["ticker", "title", "publisher", "published_utc"]
                    extra_cols = [c for c in df_urls.columns if c not in base_cols]
                    st.session_state["input_df"] = df_urls[base_cols + extra_cols]
                    st.success(f"Loaded {len(df_urls)} articles. You can edit them below.")

    with tab_manual:
        st.subheader("Edit input rows manually")
        st.markdown(
            """
            Each row should have:
            - **ticker** (e.g. NVDA)  
            - **title** (headline text)  
            - **publisher** (Reuters, Yahoo Finance, etc.)  
            - **published_utc** (ISO, e.g. `2025-11-13T13:00:00Z`)

            You can add or delete rows as needed.
            """
        )
        # Main editor is shared below.

    # --- Unified editor: whatever is in session_state["input_df"] ---
    st.subheader("📝 Current inputs (editable)")
    edited_df = st.data_editor(
        st.session_state["input_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="input_editor",
    )
    st.session_state["input_df"] = edited_df

    # --- Run pipeline button ---
    run_button = st.button("🚀 Run model & rank signals", type="primary")

    if run_button:
        # Clean & validate
        input_df = clean_input_df(edited_df)
        if input_df.empty:
            st.error("You must provide at least one row with a ticker and a title.")
            st.stop()

        # Check scripts exist
        if not PREDICT_SCRIPT.exists():
            st.error(f"predict.py not found at: {PREDICT_SCRIPT}")
            st.stop()
        if not SCORE_SCRIPT.exists():
            st.error(f"score_signals_with_events.py not found at: {SCORE_SCRIPT}")
            st.stop()

        with st.spinner("Running model pipeline (predict → score_signals_with_events)…"):
            # Create temp files in the same folder (so relative paths work)
            tmp_in = tempfile.NamedTemporaryFile(
                dir=BASE_DIR, suffix="_fresh_input.csv", delete=False
            )
            tmp_scored = tempfile.NamedTemporaryFile(
                dir=BASE_DIR, suffix="_scored.csv", delete=False
            )
            tmp_ranked = tempfile.NamedTemporaryFile(
                dir=BASE_DIR, suffix="_ranked.csv", delete=False
            )

            tmp_in_path = Path(tmp_in.name)
            tmp_scored_path = Path(tmp_scored.name)
            tmp_ranked_path = Path(tmp_ranked.name)

            # We don't need the open file handles
            tmp_in.close()
            tmp_scored.close()
            tmp_ranked.close()

            try:
                # 1) Write input
                input_df[["ticker", "title", "publisher", "published_utc"]].to_csv(
                    tmp_in_path, index=False
                )

                # 2) Call predict.py
                ok1, out1, err1 = run_subprocess(
                    [
                        str(PREDICT_SCRIPT),
                        "--input",
                        str(tmp_in_path),
                        "--output",
                        str(tmp_scored_path),
                    ]
                )
                if not ok1:
                    st.error("Error running predict.py")
                    with st.expander("predict.py stdout/stderr"):
                        st.code(out1 or "(no stdout)", language="text")
                        st.code(err1 or "(no stderr)", language="text")
                    st.stop()

                # 3) Call score_signals_with_events.py
                ok2, out2, err2 = run_subprocess(
                    [
                        str(SCORE_SCRIPT),
                        "--input",
                        str(tmp_scored_path),
                        "--output",
                        str(tmp_ranked_path),
                        "--top",
                        str(top_fraction),
                    ]
                )
                if not ok2:
                    st.error("Error running score_signals_with_events.py")
                    with st.expander("score_signals_with_events.py stdout/stderr"):
                        st.code(out2 or "(no stdout)", language="text")
                        st.code(err2 or "(no stderr)", language="text")
                    st.stop()

                # 4) Load ranked results
                ranked_df = pd.read_csv(tmp_ranked_path)

            finally:
                # Clean up temp files (optional)
                try:
                    tmp_in_path.unlink(missing_ok=True)
                    tmp_scored_path.unlink(missing_ok=True)
                    tmp_ranked_path.unlink(missing_ok=True)
                except Exception:
                    pass

        st.success("Done! Showing ranked signals below.")

        # --- Show results ---
        st.subheader("📊 Ranked signals")

        # Collapsible: explain columns & calculations
        with st.expander("❓ What do these columns mean?"):
            st.markdown(
                """
                Below is what each key column represents and how it's calculated:

                - **tier**  
                  Discrete bucket for signal strength.  
                  • **A** = top tier (highest `signal_score`)  
                  • **B** = medium strength  
                  • **C** = weaker / more ambiguous signals  

                - **signal_score**  
                  Final composite score for this headline.  
                  It combines:
                  * `model_component` → how much the ML model's 10-day up probability is above a baseline  
                  * `event_component` → how historically strong this `event_type` has been over 10 days  
                  Both are adjusted by publisher and confidence weights. Higher = stronger bullish setup.

                - **prob_up_10d**  
                  Raw model output: estimated probability (0–1) that the stock's 10-day forward return is positive,
                  given this headline and the structured features it sees.

                - **event_type**  
                  Rule-based label from `event_types.py` that categorizes the headline. Examples:  
                  `contract_major`, `contract_win`, `earnings_miss_or_cut`, `regulatory_approval`, `legal_risk`, `product_launch`, `other`.  
                  This is used to look up backtested stats for similar events.

                - **base_prob**  
                  Baseline probability that similar events finished up over 10 days.  
                  For a given `event_type`, this is usually:  
                  `base_prob = historical hit_rate_10d for that event_type in the training backtest`  
                  If there’s not enough data, it falls back toward the global baseline.

                - **win_edge**  
                  Extra edge from the model vs. baseline:  
                  `win_edge = prob_up_10d - base_prob`  
                  If positive → model thinks this specific headline is better than the average outcome for its event_type.

                - **event_edge**  
                  Historical drift of the event itself:  
                  `event_edge = avg_10d` from the backtest for this `event_type`.  
                  Positive means this type of event has historically led to positive 10-day returns on average.

                - **conf_weight**  
                  Confidence multiplier based on sample size for that `event_type`.  
                  * Many past examples → weight closer to 1  
                  * Very few examples → weight shrinks toward 0 so you don’t over-trust noisy event stats.

                - **pub_weight**  
                  Manual quality weight for the news source.  
                  * High-tier outlets (e.g. Reuters, Bloomberg) → slightly above 1  
                  * Neutral sources → ≈ 1  
                  * Lower-quality / noisy sources → can be below 1  
                  This only affects the scoring components, not the raw `prob_up_10d`.

                - **model_component**  
                  Part of the score driven by the ML model.  
                  Roughly: positive portion of `(prob_up_10d - base_prob)` scaled by `pub_weight`.  
                  Captures how much this specific headline looks better than a “typical” event of the same type.

                - **event_component**  
                  Part of the score driven by backtested event behavior.  
                  Roughly: `event_edge * conf_weight`, then blended with `pub_weight`.  
                  Captures how strong this event_type has been historically, adjusted for sample size and source.

                - **reason**  
                  Human-readable explanation string that summarizes why the headline got this score, including:  
                  * `event_type`  
                  * historical 10-day stats for that event_type  
                  * `prob_up_10d` from the model  
                  * `pub_w` (publisher weight)
                """
            )

        # Prioritize the most useful columns
        preferred_cols = [
            "tier",
            "signal_score",
            "ticker",
            "title",
            "publisher",
            "published_utc",
            "event_type",
            "prob_up_10d",
            "reason",
        ]
        cols_in_df = [c for c in preferred_cols if c in ranked_df.columns]
        other_cols = [c for c in ranked_df.columns if c not in cols_in_df]
        display_df = ranked_df[cols_in_df + other_cols]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv_bytes = ranked_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download ranked CSV",
            data=csv_bytes,
            file_name="ranked_signals.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
