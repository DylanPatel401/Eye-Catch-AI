# pip install pandas requests python-dateutil
import os, sys, time, argparse, json
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN", "").strip()
if not FINNHUB_TOKEN:
    sys.exit(
        'ERROR: Set FINNHUB_TOKEN (PowerShell):  setx FINNHUB_TOKEN "your_key"  and reopen terminal.'
    )

# === NEW ALPHA UNIVERSE (replaces old HIGH_CATALYST_30 mega-caps) ===
# Small/mid-cap, headline- and insider-sensitive names
HIGH_CATALYST_40 = [
    # Biotech / pharma
    "VERA", "KRTX", "IONS", "EXAI", "RNA", "BEAM", "EDIT", "VERV", "ALLO", "ARQT",
    "SRPT", "SAGE", "NVAX", "REGN",

    # SaaS / cloud / software
    "ASAN", "ESTC", "BILL", "DOCS", "PD", "FSLY", "TWOU", "PUBM", "SHOP", "ZI",

    # Clean energy / EV / solar
    "RUN", "SPWR", "BLNK", "CHPT", "PLUG", "BE", "ENVX", "MAXN",

    # Fintech / consumer / growth
    "AFRM", "UPST", "SOFI", "BMBL", "ROKU", "CELH",

    # Space / speculative tech
    "ASTS", "RKLB", "ACHR", "JOBY",
]

# Keep utilities only if you still want this alternate universe.
UTILITIES_20 = [
    "NEE", "DUK", "D", "SO", "AEP", "EXC", "EIX", "PNW", "ED", "XEL",
    "CNP", "AEE", "PPL", "AES", "PEG", "NI", "CMS", "ETR", "FE", "LNT",
]

KEYWORD_GROUPS = {
    "partnership": ["partner", "partnership", "collaboration", "joint venture", "agreement", "alliance"],
    "invests": ["invest", "investment", "funding", "stake"],
    "acquisition": ["acquire", "acquisition", "merger", "buyout", "takeover"],
    "contract": ["contract", "secures contract", "wins contract", "award", "awarded"],
    "guidance": ["guidance", "raises guidance", "lowers guidance", "outlook"],
    "lawsuit": ["lawsuit", "sues", "settlement", "probe", "investigation", "subpoena"],
    "recalls": ["recall", "defect", "safety issue"],
    "rating": ["upgrade", "downgrade", "initiated coverage"],
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


def parse_args():
    p = argparse.ArgumentParser(description="Finnhub news pull (resumable).")
    p.add_argument(
        "--universe",
        choices=["high", "utilities"],
        default="high",
        help="Which preset ticker universe to use. 'high' = alpha small/mid-cap universe, 'utilities' = defensive names.",
    )
    p.add_argument(
        "--from",
        dest="from_date",
        default=None,
        help="YYYY-MM-DD (default: 365 days ago)",
    )
    p.add_argument(
        "--to",
        dest="to_date",
        default=None,
        help="YYYY-MM-DD (default: today UTC)",
    )
    p.add_argument(
        "--outdir",
        default="data/news_raw",
        help="Output dir per-ticker CSVs.",
    )
    p.add_argument(
        "--filter-keywords",
        action="store_true",
        help="Apply catalyst keyword filter.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between tickers (rarely needed).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Use the new universe by default
    if args.universe == "high":
        tickers = HIGH_CATALYST_40
    else:
        tickers = UTILITIES_20

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=365)).isoformat() if not args.from_date else args.from_date
    end = today.isoformat() if not args.to_date else args.to_date

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
                rows.append(
                    {
                        "ticker": t,
                        "published_utc": dt,
                        "title": title,
                        "publisher": it.get("source") or "",
                        "url": it.get("url") or "",
                    }
                )
                kept += 1

            df = pd.DataFrame(rows)
            if not df.empty:
                df.drop_duplicates(
                    subset=["ticker", "title", "published_utc"], inplace=True
                )
                df.sort_values(
                    ["ticker", "published_utc"], ascending=[True, False], inplace=True
                )
                df.reset_index(drop=True, inplace=True)
            df.to_csv(out_path, index=False)
            print(f"   saved {kept} → {out_path}")
            completed += 1
            if args.delay:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Progress preserved. Already-saved tickers won’t be re-fetched on next run."
        )
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}. Progress preserved.")
    finally:
        print(f"Done. {completed} files written (check {outdir}).")


if __name__ == "__main__":
    main()
