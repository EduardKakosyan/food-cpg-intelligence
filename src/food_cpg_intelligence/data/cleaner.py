"""Text cleaning and normalization for ingested content."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from food_cpg_intelligence.data.models import Newsletter, NewsletterSection

# Smart quotes and typography replacements
_TYPOGRAPHY_MAP: dict[str, str] = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2013": "-",  # en-dash
    "\u2014": "--",  # em-dash
    "\u2026": "...",  # ellipsis
    "\xa0": " ",  # non-breaking space
}

_MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://[^\s)\"']+")


def _strip_utm_from_url(match: re.Match[str]) -> str:
    """Remove utm_* query parameters from a URL, preserving other params."""
    url = match.group(0)
    parts = urlsplit(url)
    if not parts.query:
        return url
    clean_qs = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")])
    return urlunsplit(parts._replace(query=clean_qs))


def clean_text(raw: str) -> str:
    """Apply a cleaning pipeline to raw text.

    Steps:
        1. Unicode NFKC normalization
        2. Replace smart quotes/typography with ASCII equivalents
        3. Strip URL tracking parameters (utm_*)
        4. Collapse excessive whitespace
        5. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFKC", raw)

    for old, new in _TYPOGRAPHY_MAP.items():
        text = text.replace(old, new)

    text = _URL_RE.sub(_strip_utm_from_url, text)

    # Collapse horizontal whitespace (preserve newlines)
    text = _MULTI_WHITESPACE_RE.sub(" ", text)

    # Collapse excessive blank lines to double newline
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def clean_newsletter(newsletter: Newsletter) -> Newsletter:
    """Apply text cleaning to all text fields of a Newsletter."""
    cleaned_sections = tuple(
        NewsletterSection(
            heading=clean_text(s.heading),
            body=clean_text(s.body),
            section_type=s.section_type,
        )
        for s in newsletter.sections
    )
    return Newsletter(
        blog_number=newsletter.blog_number,
        title=clean_text(newsletter.title),
        published_date=newsletter.published_date,
        author=newsletter.author,
        sections=cleaned_sections,
        raw_word_count=newsletter.raw_word_count,
        source_file=newsletter.source_file,
        channel=newsletter.channel,
    )
