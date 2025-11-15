# event_types.py
#
# Rule-based event classifier for headlines.
# Maps a news title to one of a small set of event_type strings.
# This version is stricter and tries to avoid over-tagging everything
# as "contract_win" / "other", and adds a "contract_major" bucket.

from typing import Optional


def _contains_any(t: str, words) -> bool:
    return any(w in t for w in words)


def classify_event(title: Optional[str]) -> str:
    if not isinstance(title, str) or not title.strip():
        return "other"

    t = title.lower()

    # =========================
    # 1. HARD NEGATIVE EVENTS
    # =========================

    # Legal / regulatory risk
    if _contains_any(t, [
        "lawsuit",
        "sues ",
        "sued ",
        "class action",
        "subpoena",
        "investigation",
        "probe",
        "sec charges",
        "sec complaint",
        "doj charges",
        "antitrust",
        "fraud allegations",
        "bribery",
        "fine from",
        "regulator fines",
        "penalty",
        "settles with sec",
        "settles with doj",
    ]):
        return "legal_risk"

    # Explicit negative moves (price action headlines)
    if _contains_any(t, [
        "slump", "slumps", "plunge", "plunges", "tumbles", "falls",
        "sinks", "slides", "selloff", "dives", "tanks", "crashes",
        "hits 52-week low", "hits 52 week low",
    ]):
        return "negative_move"

    # Operational / safety / recall / outage
    if _contains_any(t, [
        "production halt",
        "halts production",
        "suspends production",
        "suspends operations",
        "factory shutdown",
        "plant shutdown",
        "shuts plant",
        "shutdown of",
        "recall",
        "defect",
        "safety issue",
        "safety concerns",
        "grounded",
        "data breach",
        "security breach",
        "ransomware",
        "outage",
        "service disruption",
        "accident",
        "crash",
        "explosion",
        "fire at",
    ]):
        return "operational_issue"

    # Earnings miss or guidance cut
    if _contains_any(t, [
        "misses expectations",
        "miss expectations",
        "miss estimates",
        "missed estimates",
        "revenue miss",
        "eps miss",
        "results miss",
        "falls short of estimates",
        "below estimates",
        "below expectations",
        "miss on revenue",
        "misses on revenue",
        "misses on eps",
    ]):
        return "earnings_miss_or_cut"

    if _contains_any(t, [
        "cuts guidance",
        "cuts its guidance",
        "lowers guidance",
        "trims guidance",
        "guidance cut",
        "cuts outlook",
        "lowers outlook",
        "warns on revenue",
        "warns on outlook",
        "issues profit warning",
        "warning on revenue",
        "warns of slowdown",
        "issues weak outlook",
        "issues cautious outlook",
    ]):
        return "earnings_miss_or_cut"

    # Regulatory "rejection" style items (treated like negative fundamental)
    if _contains_any(t, [
        "complete response letter",
        "crl from fda",
        "fda rejects",
        "fda declines",
        "rejects application",
        "refuses approval",
        "refusal to file",
    ]):
        return "earnings_miss_or_cut"

    # Analyst downgrade (keep separate from other negatives)
    if _contains_any(t, [
        "downgrade to",
        "downgraded to",
        "cut to sell",
        "cut to hold",
        "cut to neutral",
        "downgrades shares",
        "downgrades stock",
        "cuts rating",
        "slashed rating",
        "slashes rating",
    ]):
        return "analyst_downgrade"

    # =========================
    # 2. STRONG POSITIVE EVENTS
    # =========================

    # M&A / takeovers (do this before "deal/contract")
    if _contains_any(t, [
        "acquire ",
        "acquisition of",
        "agrees to acquire",
        "to acquire",
        "buyout",
        "takeover",
        "merger",
        "merges with",
        "to combine with",
        "to be bought by",
        "agrees to buy",
        "to buy rival",
    ]):
        return "m&a"

    # Regulatory approvals
    if _contains_any(t, [
        "fda approves",
        "fda approval",
        "wins fda approval",
        "regulator approves",
        "approval from regulator",
        "regulatory approval",
        "grants approval",
        "receives approval",
        "wins approval",
        "cleared by fda",
        "gets approval",
        "ec approval",
        "ema approval",
    ]):
        return "regulatory_approval"

    # Earnings beat
    if _contains_any(t, [
        "beats expectations",
        "beat expectations",
        "tops expectations",
        "above expectations",
        "beats estimates",
        "beat estimates",
        "tops estimates",
        "crushes estimates",
        "blows past estimates",
        "results beat",
        "earnings beat",
        "smashes estimates",
        "surpasses estimates",
    ]):
        return "earnings_beat"

    # Guidance raise (separate from beat)
    if _contains_any(t, [
        "raises guidance",
        "hikes guidance",
        "boosts guidance",
        "raises outlook",
        "lifts outlook",
        "improves outlook",
        "raises forecast",
        "boosts forecast",
        "lifts forecast",
        "increases forecast",
    ]):
        return "guidance_raise"

    # Buyback / capital return
    if _contains_any(t, [
        "buyback",
        "repurchase program",
        "share repurchase",
        "stock repurchase",
        "authorizes buyback",
        "announces buyback",
        "announces repurchase",
        "boosts buyback",
        "increases buyback",
    ]):
        return "buyback"

    # Analyst upgrades (positive side)
    if _contains_any(t, [
        "upgrade to buy",
        "upgraded to buy",
        "upgrade to overweight",
        "upgraded to overweight",
        "upgrade to outperform",
        "upgraded to outperform",
        "upgrades shares",
        "upgrades stock",
        "initiated with buy",
        "initiates with buy",
        "initiates with overweight",
        "initiates with outperform",
        "price target raised",
        "raises price target",
    ]):
        return "analyst_upgrade"

    # Insider activity
    if _contains_any(t, [
        "insider buys",
        "insider purchase",
        "director buys",
        "ceo buys",
        "cfo buys",
        "buys stock",
        "buys shares",
        "purchases shares",
        "acquires shares",
        "buys more shares",
    ]):
        return "insider_buy"

    if _contains_any(t, [
        "insider sells",
        "insider selling",
        "director sells",
        "ceo sells",
        "cfo sells",
        "sells stock",
        "sells shares",
        "dumps shares",
        "disposes shares",
    ]):
        return "insider_sell"

    # ====== CONTRACTS: MAJOR vs NORMAL ======

    # Major contract wins (government / multi-billion / huge)
    if (
        ("contract" in t or "order" in t or "tender" in t)
        and _contains_any(t, [
            "wins ", "win ", "awarded", "secures", "secured",
            "lands", "books", "receives order", "receives contract",
            "selected by", "chosen by", "inked", "signs", "signed",
        ])
        and _contains_any(t, [
            "billion",
            "multi-billion",
            "multibillion",
            "$1b", "$2b", "$3b", "$4b", "$5b", "$10b",
            "government",
            "u.s. government",
            "us government",
            "pentagon",
            "defense department",
            "department of defense",
            "u.s. army",
            "us army",
            "air force",
            "navy",
            "nasa",
        ])
    ):
        return "contract_major"

    # Normal contract wins
    if (
        "contract" in t or "order" in t or "tender" in t
    ) and _contains_any(t, [
        "wins ",
        "win ",
        "awarded",
        "secures",
        "secured",
        "lands",
        "books",
        "receives order",
        "receives contract",
        "selected by",
        "chosen by",
        "inked",
        "signs",
        "signed",
    ]):
        return "contract_win"

    # Partnership / strategic alliance
    if _contains_any(t, [
        "partners with",
        "partnership with",
        "enters partnership",
        "strategic partnership",
        "collaboration with",
        "enters collaboration",
        "teams up with",
        "alliance with",
        "strategic alliance",
        "joint venture with",
        "forms joint venture",
    ]):
        return "partnership"

    # Product / service launches
    if _contains_any(t, [
        "launches ",
        "launch of ",
        "unveils ",
        "introduces ",
        "debuts ",
        "rolls out",
        "new product",
        "new service",
        "new platform",
        "new chip",
        "new ai model",
        "new feature",
    ]):
        return "product_launch"

    # =========================
    # 3. SENTIMENT-ONLY PRICE MOVE (IF STILL NOT CAUGHT)
    # =========================

    if _contains_any(t, [
        "surges",
        "jumps",
        "soars",
        "rallies",
        "spikes",
        "climbs",
        "leaps",
        "spikes higher",
        "hits 52-week high",
        "hits 52 week high",
    ]):
        return "positive_move"

    # =========================
    # 4. FALLBACK
    # =========================

    return "other"
