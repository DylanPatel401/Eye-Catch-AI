# event_types.py
#
# Simple rule-based event classifier for headlines.
# Maps a news title to one of a small set of event_type strings.
# You can expand / tweak these rules later.

from typing import Optional

def classify_event(title: Optional[str]) -> str:
    if not isinstance(title, str) or not title.strip():
        return "other"

    t = title.lower()

    # --- Strong negative events first ---
    if any(w in t for w in ["lawsuit", "sues", "subpoena", "investigation", "probe", "fraud", "sec charges"]):
        return "legal_risk"

    if any(w in t for w in ["downgrade", "downgraded", "cut to", "cut rating", "slashed rating"]):
        return "analyst_downgrade"

    if any(w in t for w in [
        "misses expectations", "miss expectations", "revenue miss", "eps miss",
        "below estimates", "miss on", "misses on"
    ]):
        return "earnings_miss_or_cut"

    if any(w in t for w in [
        "cuts guidance", "lowers guidance", "trims guidance", "guidance cut",
        "warns on revenue", "warning on revenue", "warns on outlook"
    ]):
        return "earnings_miss_or_cut"

    if any(w in t for w in [
        "production halt", "halts production", "production stopped",
        "safety issue", "recall", "defect", "accident", "crash", "explosion"
    ]):
        return "operational_issue"

    # --- Strong positive events ---
    if any(w in t for w in ["acquire", "acquisition", "buyout", "takeover", "merger"]):
        return "m&a"

    if any(w in t for w in ["contract", "award", "order", "deal", "agreement", "tender"]):
        return "contract_win"

    if any(w in t for w in [
        "raises guidance", "hikes guidance", "boosts guidance",
        "raises outlook", "lifts outlook", "improves outlook"
    ]):
        return "guidance_raise"

    if any(w in t for w in [
        "beats expectations", "beat expectations", "above estimates",
        "tops estimates", "crushes estimates", "blows past estimates"
    ]):
        return "earnings_beat"

    if any(w in t for w in ["buyback", "repurchase program", "share repurchase", "authorizes buyback"]):
        return "buyback"

    # --- Insider behavior ---
    if any(w in t for w in [
        "insider buys", "insider purchase", "director buys", "ceo buys",
        "buys stock", "purchases shares", "acquires shares"
    ]):
        return "insider_buy"

    if any(w in t for w in [
        "insider sells", "director sells", "ceo sells",
        "sells stock", "dumps shares", "disposes shares"
    ]):
        return "insider_sell"

    # --- Analyst actions (positive side) ---
    if any(w in t for w in [
        "upgrade", "upgraded to", "initiated with buy",
        "initiates with buy", "overweight", "outperform"
    ]):
        return "analyst_upgrade"

    # --- Other positive events ---
    if any(w in t for w in [
        "partnership", "partner", "collaboration", "alliance"
    ]):
        return "partnership"

    if any(w in t for w in [
        "launches", "unveils", "introduces", "debuts", "rolls out", "new product", "new service"
    ]):
        return "product_launch"

    if any(w in t for w in [
        "fda approves", "fda approval", "regulator approves", "approval from regulator",
        "regulatory approval", "wins approval"
    ]):
        return "regulatory_approval"

    # --- Sentimenty price moves (if explicitly mentioned) ---
    if any(w in t for w in [
        "slump", "plunge", "tumbles", "falls", "sinks", "slides",
        "slumps", "selloff", "dives", "tanks"
    ]):
        return "negative_move"

    if any(w in t for w in [
        "surges", "jumps", "soars", "rallies", "spikes", "climbs", "leaps", "spikes higher"
    ]):
        return "positive_move"

    # Fallback
    return "other"
