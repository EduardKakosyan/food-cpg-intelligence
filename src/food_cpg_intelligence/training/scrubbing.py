"""Entity scrubbing for SFT training data.

Per the voice-fine-tune research synthesis (docs/research-voice-rag-pivot.md, §4),
aggressive entity scrubbing is the single biggest lever for keeping facts out
of LoRA weights. Specific brand names, retailer names, dollar amounts, dates,
and percentages get replaced with stable placeholders. The model then learns
*how* Peter explains things, not *what* the specific numbers were.

Placeholders we emit:
    [brand], [retailer], [money], [percent], [year], [date], [number]

The scrubber is intentionally regex-based — fast, deterministic, debuggable.
Brand and retailer lists are curated below from the actual SKUFood corpus.
Add entries as we discover them in the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canadian grocery retailers and major Peter-corpus brands. Order matters:
# multi-word names must come before single-word substrings (Loblaws before Loblaw).
_RETAILERS_RAW: tuple[str, ...] = (
    "Loblaws",
    "Loblaw",
    "Sobeys",
    "Sobey's",
    "Metro Inc",
    "Walmart Canada",
    "Walmart",
    "Costco Wholesale",
    "Costco",
    "Whole Foods Market",
    "Whole Foods",
    "Empire Company",
    "Empire",
    "Save-On-Foods",
    "Save On Foods",
    "Save-On",
    "Real Canadian Superstore",
    "Superstore",
    "Federated Co-operatives",
    "Federated Co-op",
    "Federated Coop",
    "FCL",
    "Longo's",
    "Longos",
    "Farm Boy",
    "Calgary Co-op",
    "T&T Supermarket",
    "T & T",
    "Galen Weston",
    "George Weston",
    "Choices Markets",
    "Pusateri's",
    "IGA",
    "Foodland",
    "Co-op",
)

_BRANDS_RAW: tuple[str, ...] = (
    "Kirkland Signature",
    "Kirkland",
    "President's Choice",
    "PC Optimum",
    "Compliments",
    "Sensations",
    "Selection",
    "Great Value",
    "No Name",
    "Heinz",
    "Kraft",
    "Maple Leaf",
    "Schneiders",
    "Saputo",
    "Lassonde",
    "Olymel",
    "Cavendish Farms",
    "McCain",
)


@dataclass(frozen=True)
class ScrubReport:
    """Audit of one scrub pass — counts per placeholder type."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_replacements(self) -> int:
        return sum(self.counts.values())


def _word_boundary_pattern(literal: str) -> re.Pattern[str]:
    """Compile a case-insensitive whole-word matcher for a literal phrase.

    Handles internal apostrophes and hyphens (Sobey's, Save-On) by escaping.
    """
    escaped = re.escape(literal)
    # `\b` doesn't fire next to apostrophes/hyphens reliably; use lookarounds
    # over alphanumeric/underscore so abuts like "Loblaws's" still match.
    return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)


_RETAILER_PATTERNS = [(_word_boundary_pattern(r), r) for r in _RETAILERS_RAW]
_BRAND_PATTERNS = [(_word_boundary_pattern(b), b) for b in _BRANDS_RAW]


# Money: $1, $1.99, $1,000, $1.5M, $2 billion
_MONEY_RE = re.compile(
    r"""\$\s*\d[\d,]*(?:\.\d+)?\s*(?:[KkMmBb]\b|million|billion|thousand)?""",
    re.VERBOSE,
)

# Percentages: 5%, 5.5%, 5 percent
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)

# Years 1900-2099, optional 's suffix (2020s, 2022's)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}(?:'?s)?\b")

# Dates: Jan 15, 2024 / January 15 / 15 January 2024 / 2024-01-15 / 1/15/2024
_MONTH = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_RE = re.compile(
    rf"""(?:
        \b(?:{_MONTH})\s+\d{{1,2}}(?:,\s*\d{{4}})?\b   # Jan 15, 2024
      | \b\d{{1,2}}\s+(?:{_MONTH})(?:\s+\d{{4}})?\b   # 15 January 2024
      | \b\d{{4}}-\d{{2}}-\d{{2}}\b                    # 2024-01-15
      | \b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b              # 1/15/2024
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Standalone large numbers (4+ digits or with commas / units), but NOT years.
# Catches: 1,000 / 50000 / 2.5x / 1500 units. Skip 1-3 digit ints to keep
# small Peter-isms like "3 SKUs" or "6 sessions" readable in training.
_BIG_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{4,}(?:\.\d+)?\b")


def _sub_counting(pattern: re.Pattern[str], placeholder: str, text: str) -> tuple[str, int]:
    """re.subn but skip when no match — keeps counts honest."""
    new_text, count = pattern.subn(placeholder, text)
    return new_text, count


def scrub_text(raw: str) -> ScrubReport:
    """Run the full scrubbing pipeline on a single string.

    Order matters: dates/years run before bare numbers so they're not
    double-counted; named entities run before catch-all big-number regex.
    """
    counts: dict[str, int] = {}
    text = raw

    for pattern, _name in _RETAILER_PATTERNS:
        text, c = _sub_counting(pattern, "[retailer]", text)
        if c:
            counts["[retailer]"] = counts.get("[retailer]", 0) + c

    for pattern, _name in _BRAND_PATTERNS:
        text, c = _sub_counting(pattern, "[brand]", text)
        if c:
            counts["[brand]"] = counts.get("[brand]", 0) + c

    text, c = _sub_counting(_DATE_RE, "[date]", text)
    if c:
        counts["[date]"] = c

    text, c = _sub_counting(_YEAR_RE, "[year]", text)
    if c:
        counts["[year]"] = c

    text, c = _sub_counting(_MONEY_RE, "[money]", text)
    if c:
        counts["[money]"] = c

    text, c = _sub_counting(_PERCENT_RE, "[percent]", text)
    if c:
        counts["[percent]"] = c

    text, c = _sub_counting(_BIG_NUMBER_RE, "[number]", text)
    if c:
        counts["[number]"] = c

    return ScrubReport(text=text, counts=counts)
