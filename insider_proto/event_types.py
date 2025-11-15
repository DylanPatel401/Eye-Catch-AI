# event_types.py
#
# Rule-based event classifier for headlines.
# Maps a news title to one of a small set of event_type strings.
#
# Goals:
# - Catch truly catastrophic negatives (bankruptcy, default, delisting, fraud).
# - Distinguish strong fundamental events (M&A, big contracts, approvals, etc.).
# - Avoid over-tagging everything as contract_win/other.
# - Keep labels mostly aligned with what your backtests already used
#   so event stats remain meaningful.

from typing import Optional, Iterable


def _contains_any(t: str, phrases: Iterable[str]) -> bool:
    """Return True if any phrase is a substring of t (case already lowered)."""
    return any(p in t for p in phrases)


def classify_event(title: Optional[str]) -> str:
    if not isinstance(title, str) or not title.strip():
        return "other"

    t = title.lower().strip()

    # ==========================================================
    # 0. CATASTROPHIC FUNDAMENTAL RISK (hard negative categories)
    # ==========================================================

    # Bankruptcy / insolvency / going concern / delisting / default
    if _contains_any(t, [
        "files for bankruptcy",
        "filed for bankruptcy",
        "seeks bankruptcy protection",
        "chapter 11",
        "chapter 7 bankruptcy",
        "chapter 15 bankruptcy",
        "insolvency",
        "insolvent",
        "liquidation",
        "to liquidate",
        "winding down operations",
        "wind down operations",
        "shuts down operations",
        "ceases operations",
        "go out of business",
        "going concern warning",
        "substantial doubt about its ability to continue as a going concern",
        "substantial doubt about its ability to continue",
        "may not be able to continue as a going concern",
        "delisting notice",
        "to be delisted",
        "faces delisting",
        "removed from nasdaq",
        "removed from nyse",
        "defaults on debt",
        "payment default",
        "bond default",
        "loan default",
        "misses bond payment",
        "misses interest payment",
    ]):
        return "bankruptcy"  # new, treated as catastrophic in scoring

    # Accounting fraud / restatement / serious misconduct
    if _contains_any(t, [
        "accounting irregularities",
        "accounting scandal",
        "restates earnings",
        "restates results",
        "restatement of financial",
        "material weakness in internal controls",
        "material weakness in internal control",
        "fraud investigation",
        "allegations of fraud",
        "charged with fraud",
        "sec fraud charges",
        "sec charges company",
        "whistleblower complaint",
        "kickback scheme",
        "bribery scandal",
    ]):
        return "fraud_or_misconduct"  # new, strong structural negative

    # ==========================================================
    # 1. LEGAL, REGULATORY, OPERATIONAL, EARNINGS NEGATIVES
    # ==========================================================

    # Legal / regulatory risk (lawsuits, investigations, regulators)
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
        "sec sues",
        "doj charges",
        "justice department charges",
        "antitrust case",
        "antitrust lawsuit",
        "fraud allegations",
        "patent infringement",
        "patent suit",
        "patent dispute",
        "patent case",
        "must pay damages",
        "ordered to pay damages",
        "regulator fines",
        "regulatory fine",
        "penalty from regulator",
        "settles with sec",
        "settles with doj",
        "settlement with regulator",
    ]):
        return "legal_risk"

    # Explicit negative price action headlines
    if _contains_any(t, [
        "shares slump",
        "stock slumps",
        "slump ",
        "slumps ",
        "plunge",
        "plunges",
        "tumbles",
        "falls ",
        "sinks",
        "slides",
        "selloff",
        "sell-off",
        "dives",
        "tanks",
        "crashes",
        "hits 52-week low",
        "hits 52 week low",
        "hits record low",
    ]):
        return "negative_move"

    # Operational / safety / recall / outage
    if _contains_any(t, [
        "production halt",
        "halts production",
        "production stopped",
        "suspends production",
        "suspends operations",
        "factory shutdown",
        "plant shutdown",
        "shuts plant",
        "shutdown of",
        "recall",
        "product recall",
        "safety issue",
        "safety concerns",
        "quality issue",
        "quality problems",
        "grounded fleet",
        "grounding its fleet",
        "data breach",
        "security breach",
        "ransomware attack",
        "cyberattack",
        "outage",
        "service disruption",
        "system outage",
        "accident",
        "crash",
        "explosion",
        "fire at",
    ]):
        return "operational_issue"

    # Earnings miss (numbers below expectations)
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
        "profit miss",
        "earnings miss",
    ]):
        return "earnings_miss_or_cut"

    # Guidance cut / weak outlook
    if _contains_any(t, [
        "cuts guidance",
        "cuts its guidance",
        "lowers guidance",
        "trims guidance",
        "guidance cut",
        "cuts outlook",
        "lowers outlook",
        "trims outlook",
        "warns on revenue",
        "warns on outlook",
        "issues profit warning",
        "profit warning",
        "warning on revenue",
        "warns of slowdown",
        "issues weak outlook",
        "issues cautious outlook",
    ]):
        return "earnings_miss_or_cut"

    # Regulatory rejection style items (CRL, etc.) – keep mapped to same bucket
    # as in your original backtest: earnings_miss_or_cut
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

    # Analyst downgrade (rating or target lowered)
    if _contains_any(t, [
        "downgrade to",
        "downgraded to",
        "downgrades shares",
        "downgrades stock",
        "cuts rating to",
        "slashed rating",
        "slashes rating",
        "cut to sell",
        "cut to hold",
        "cut to neutral",
        "lowered to hold",
        "lowered to sell",
    ]):
        return "analyst_downgrade"

    # ==========================================================
    # 2. STRONG FUNDAMENTAL POSITIVES
    # ==========================================================

    # M&A / takeovers (run before generic "deal/contract" logic)
    if _contains_any(t, [
        "acquire ",
        "acquisition of",
        "agrees to acquire",
        "to acquire ",
        "buyout",
        "takeover",
        "merger",
        "merges with",
        "to combine with",
        "to be bought by",
        "agrees to buy",
        "to buy rival",
        "take-private deal",
        "take private deal",
    ]):
        return "m&a"

    # Regulatory approvals (keep as separate positive bucket)
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
        "marketing authorization",
    ]):
        return "regulatory_approval"

    # Earnings beat (results above expectations)
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

    # Guidance raise
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

    # Buybacks / capital return
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
        "expands buyback",
        "special dividend",
        "raises dividend",
    ]):
        return "buyback"

    # Analyst upgrades (positive side)
    if _contains_any(t, [
        "upgrade to buy",
        "upgraded to buy",
        "upgrades to buy",
        "upgrade to overweight",
        "upgraded to overweight",
        "upgrades to overweight",
        "upgrade to outperform",
        "upgraded to outperform",
        "upgrades to outperform",
        "initiated with buy",
        "initiates with buy",
        "initiates with overweight",
        "initiates with outperform",
        "rates buy",
        "rates outperform",
        "rates overweight",
        "price target raised",
        "raises price target",
        "boosts price target",
    ]):
        return "analyst_upgrade"

    # ==========================================================
    # 3. INSIDER ACTIVITY
    # ==========================================================

    # Insider buying
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
        "significant insider buying",
        "heavy insider buying",
    ]):
        return "insider_buy"

    # Insider selling
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
        "heavy insider selling",
    ]):
        return "insider_sell"

    # ==========================================================
    # 4. CONTRACTS, PARTNERSHIPS, PRODUCT LAUNCHES
    # ==========================================================

    # Major contracts only when clearly won/awarded AND big / strategic.
    # This maps to "contract_major" that you already use in your scoring.
    if (
        _contains_any(t, ["contract", "order", "deal", "agreement", "tender", "award"])
        and _contains_any(t, [
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
            "signs multi-year",
            "signed multi-year",
        ])
        and _contains_any(t, [
            "billion",
            "billion-dollar",
            "billion dollar",
            "multi-year",
            "multiyear",
            "long-term",
            "long term",
            "u.s. government",
            "us government",
            "defense department",
            "department of defense",
            "pentagon",
            "army",
            "navy",
            "air force",
        ])
    ):
        return "contract_major"

    # Regular contract wins (awarded/secured but not obviously massive)
    if (
        _contains_any(t, ["contract", "order", "deal", "agreement", "tender", "award"])
        and _contains_any(t, [
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
            "signs contract",
            "signs deal",
        ])
    ):
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
        "joint venture between",
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
        "new version of",
    ]):
        return "product_launch"

    # ==========================================================
    # 5. POSITIVE PRICE MOVES (IF NOT ALREADY TAGGED)
    # ==========================================================

    if _contains_any(t, [
        "surges",
        "jumps",
        "soars",
        "rallies",
        "spikes",
        "climbs",
        "leaps",
        "spikes higher",
        "reverses higher",
        "bounces back",
        "rebounds",
        "recovers",
        "hits 52-week high",
        "hits 52 week high",
        "hits record high",
    ]):
        return "positive_move"

    # ==========================================================
    # 6. FALLBACK
    # ==========================================================

    return "other"
