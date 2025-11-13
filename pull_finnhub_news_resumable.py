# pull_finnhub_news_resumable.py
# pip install pandas requests python-dateutil

import os, sys, time, argparse
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "").strip()
if not FINNHUB_TOKEN:
    sys.exit(
        'ERROR: Set FINNHUB_TOKEN (PowerShell):  setx FINNHUB_TOKEN "your_key"  and reopen terminal.'
    )

# Legacy presets (you can keep or ignore)
HIGH_CATALYST_40 = [
    # Biotech / pharma
    "VERA", "KRTX", "IONS", "EXAI", "RNA", "BEAM", "EDIT", "VERV", "ALLO", "ARQT",
    "SRPT", "SAGE", "NVAX", "REGN", "BMRN", "VRTX",
    # SaaS / cloud
    "ASAN", "ESTC", "BILL", "DOCS", "PD", "FSLY", "TWOU", "PUBM", "ZI", "TTD",
    # Clean energy / EV / solar
    "RUN", "SPWR", "BLNK", "CHPT", "PLUG", "BE", "ENVX", "MAXN",
    # Fintech / consumer
    "AFRM", "UPST", "SOFI", "BMBL", "ROKU", "CELH",
    # Space / speculative tech
    "ASTS", "RKLB", "ACHR", "JOBY",
]

UTILITIES_20 = [
    "NEE", "DUK", "D", "SO", "AEP", "EXC", "EIX", "PNW", "ED", "XEL",
    "CNP", "AEE", "PPL", "AES", "PEG", "NI", "CMS", "ETR", "FE", "LNT",
]

KEYWORD_GROUPS = {
    "partnership": ["partner","partnership","collaboration","joint venture","agreement","alliance"],
    "invests": ["invest","investment","funding","stake"],
    "acquisition": ["acquire","acquisition","merger","buyout","takeover"],
    "contract": ["contract","secures contract","wins contract","award","awarded"],
    "guidance": ["guidance","raises guidance","lowers guidance","outlook"],
    "lawsuit": ["lawsuit","sues","settlement","probe","investigation","subpoena"],
    "recalls": ["recall","defect","safety issue"],
    "rating": ["upgrade","downgrade","initiated coverage"],
}
KEYWORDS = sorted({kw.lower() for v in KEYWORD_GROUPS.values() for kw in v})


def matches_keywords(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(k in t for k in KEYWORDS)


def company_news(symbol: str, start: str, end: str, retries=2, backoff=1.0):
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": start, "to": end, "token": FINNHUB_TOKEN}
    for i in range(retries + 1):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429 and i < retries:
            time.sleep(backoff * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    return []


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_universe_from_file(path: str):
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line.upper())
    if not tickers:
        raise SystemExit(f"Universe file {path} contained no tickers.")
    return sorted(set(tickers))


def parse_args():
    p = argparse.ArgumentParser(description="Finnhub news pull (resumable).")
    p.add_argument("--universe", choices=["high", "utilities"], default="high",
                   help="Which preset ticker universe to use (ignored if --universe-file is set).")
    p.add_argument("--universe-file", default=None,
                   help="Path to text file with one ticker per line. Overrides --universe presets if provided.")
    p.add_argument("--from", dest="from_date", default=None,
                   help="YYYY-MM-DD (default: 5 years ago)")
    p.add_argument("--to", dest="to_date", default=None,
                   help="YYYY-MM-DD (default: today UTC)")
    p.add_argument("--outdir", default="data/news_alpha", help="Output dir per-ticker CSVs.")
    p.add_argument("--filter-keywords", action="store_true",
                   help="Apply catalyst keyword filter.")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to sleep between tickers.")
    return p.parse_args()


def main():
    args = parse_args()

    # Universe selection
    if args.universe_file:
        tickers = load_universe_from_file(args.universe_file)
        print(f"Loaded {len(tickers)} tickers from {args.universe_file}")
    else:
        tickers = HIGH_CATALYST_40 if args.universe == "high" else UTILITIES_20

    today = datetime.now(timezone.utc).date()
    default_start = today - timedelta(days=5*365)  # ~5 years

    start = default_start.isoformat() if not args.from_date else args.from_date
    end   = today.isoformat() if not args.to_date else args.to_date

    outdir = args.outdir
    ensure_dir(outdir)

    print(f"Pulling {len(tickers)} tickers  {start} → {end}")
    print(f"Keyword filter: {'ON' if args.filter_keywords else 'OFF'}  |  Output: {outdir}")

    completed = 0
    try:
        for i, t in enumerate(tickers, 1):
            out_path = os.path.join(outdir, f"{t}.csv")
            if os.path.exists(out_path):
                print(f"[{i}/{len(tickers)}] {t}  SKIP (exists)")
                if args.delay:
                    time.sleep(args.delay)
                continue

            print(f"[{i}/{len(tickers)}] {t}  fetching…")
            items = company_news(t, start, end)
            rows = []
            kept = 0
            for it in items:
                title = (it.get("headline") or "").strip()
                if not title:
                    continue
                if args.filter_keywords and not matches_keywords(title):
                    continue
                ts = int(it.get("datetime", 0))
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                rows.append({
                    "ticker": t,
                    "published_utc": dt,
                    "title": title,
                    "publisher": it.get("source") or "",
                    "url": it.get("url") or "",
                })
                kept += 1

            df = pd.DataFrame(rows)
            if not df.empty:
                df.drop_duplicates(subset=["ticker","title","published_utc"], inplace=True)
                df.sort_values(["ticker","published_utc"], ascending=[True, False], inplace=True)
                df.reset_index(drop=True, inplace=True)
            df.to_csv(out_path, index=False)
            print(f"   saved {kept} → {out_path}")
            completed += 1
            if args.delay:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nInterrupted. Progress preserved. Already-saved tickers won’t be re-fetched on next run.")
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}. Progress preserved.")
    finally:
        print(f"Done. {completed} files written (check {outdir}).")


if __name__ == "__main__":
    main()
