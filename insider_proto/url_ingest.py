# url_ingest.py
#
# Turn a list of article URLs into a DataFrame with:
# ticker, title, publisher, published_utc
# Ticker is guessed with simple heuristics and can be edited in the UI.

from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import List, Optional

import re
import pandas as pd
from newspaper import Article


def _get_publisher_from_url(url: str) -> str:
    host = urlparse(url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    # crude mapping for nice names
    if "yahoo" in host:
        return "Yahoo Finance"
    if "bloomberg" in host:
        return "Bloomberg"
    if "marketwatch" in host:
        return "MarketWatch"
    if "marketbeat" in host:
        return "MarketBeat"
    if "reuters" in host:
        return "Reuters"
    if "cnbc" in host:
        return "CNBC"
    return host or "Unknown"


_TICKER_PATTERNS = [
    # (NASDAQ: NVDA), (Nasdaq: NVDA), (NYSE: GE) etc
    re.compile(r"\((?:NASDAQ|Nasdaq|NYSE|NYSE American|AMEX|OTC)\s*:\s*([A-Z\.]{1,6})\)"),
    # ... NVDA.O, NVDA.OQ style
    re.compile(r"\b([A-Z]{1,5})\.O\b"),
    re.compile(r"\b([A-Z]{1,5})\.N\b"),
]


def _guess_ticker(text: str, title: str) -> Optional[str]:
    blob = f"{title}\n{text}"
    for pat in _TICKER_PATTERNS:
        m = pat.search(blob)
        if m:
            return m.group(1)

    # fallback: look for something like "Shares of NVDA" or "NVDA stock"
    m2 = re.search(r"shares of\s+([A-Z]{2,5})\b", blob)
    if m2:
        return m2.group(1)

    m3 = re.search(r"\b([A-Z]{2,5})\s+stock\b", blob)
    if m3:
        return m3.group(1)

    return None


def urls_to_df(urls: List[str]) -> pd.DataFrame:
    rows = []
    for url in urls:
        url = url.strip()
        if not url:
            continue

        title = ""
        published_utc = None
        text = ""

        try:
            art = Article(url)
            art.download()
            art.parse()
            title = (art.title or "").strip()
            text = art.text or ""
            pub_date = art.publish_date
        except Exception:
            pub_date = None

        if not title:
            title = url  # worst case

        if pub_date is not None:
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            published_utc = pub_date.isoformat().replace("+00:00", "Z")
        else:
            # fallback: now, in UTC
            published_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        publisher = _get_publisher_from_url(url)
        ticker = _guess_ticker(text, title) or ""

        rows.append(
            {
                "ticker": ticker,
                "title": title,
                "publisher": publisher,
                "published_utc": published_utc,
                "source_url": url,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ticker", "title", "publisher", "published_utc", "source_url"])

    return pd.DataFrame(rows)
