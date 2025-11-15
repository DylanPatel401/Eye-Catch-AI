# app.py
#
# Streamlit frontend for your Eye-Catch-AI prototype.
#
# What it does:
# - Lets you enter headlines in a simple table (ticker, title, publisher, published_utc).
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
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
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
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
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
    st.caption("Paste/edit headlines → run your XGBoost + event backtest pipeline → get ranked signals.")

    st.markdown(
        """
        **How to use this page:**
        1. Edit the table below (add / remove rows as you want).
        2. Make sure each row has: **ticker**, **title**, **publisher**, **published_utc** (ISO like `2025-11-13T13:00:00Z`).
        3. Click **“Run model & rank signals”**.
        4. Scroll to see ranked signals, or download as CSV.
        """
    )

    with st.sidebar:
        st.header("Options")
        top_fraction = st.slider(
            "Top fraction to keep (from score_signals_with_events.py)",
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
            """
        )

    # --- Editable input table ---
    st.subheader("📝 Input headlines")

    if "input_df" not in st.session_state:
        st.session_state["input_df"] = make_default_rows()

    edited_df = st.data_editor(
        st.session_state["input_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
    st.session_state["input_df"] = edited_df

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
                input_df.to_csv(tmp_in_path, index=False)

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
                # Clean up temp files (optional; comment out if you want to inspect them)
                try:
                    tmp_in_path.unlink(missing_ok=True)
                    tmp_scored_path.unlink(missing_ok=True)
                    tmp_ranked_path.unlink(missing_ok=True)
                except Exception:
                    pass

        st.success("Done! Showing ranked signals below.")

        # --- Show results ---
        st.subheader("📊 Ranked signals")

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
