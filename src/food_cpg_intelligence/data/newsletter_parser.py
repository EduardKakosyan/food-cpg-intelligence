"""Parse SKUFood newsletter .docx files into structured Newsletter objects."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog
from docx import Document

from food_cpg_intelligence.data.models import Newsletter, NewsletterSection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docx.text.paragraph import Paragraph

logger = structlog.stdlib.get_logger(__name__)

_BLOG_NUMBER_RES = (
    re.compile(r"skufood\s*blog\s*(\d+)", re.IGNORECASE),
    # Co-authored / cross-posted content: SKUFood140thegrower, GPSNewsletter58thegrower,
    # gpsnewsletter135thegrower — also Peter prose, distributed via The Grower magazine.
    re.compile(r"(?:skufood|gps\s*newsletter)\s*(\d+)\s*thegrower", re.IGNORECASE),
)


def _extract_blog_number(filename: str) -> int:
    """Extract the blog number from a newsletter filename.

    Handles inconsistent casing (SKUFoodblog100.docx, SKUfoodblog42.docx) and
    cross-channel content (SKUFood140thegrower.docx, GPSNewsletter58thegrower.docx).
    """
    for pattern in _BLOG_NUMBER_RES:
        match = pattern.search(filename)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot extract blog number from filename: {filename}")


def _is_heading(paragraph: Paragraph) -> bool:
    """Determine if a paragraph is a section heading.

    SKUFood newsletters use bold text for headings rather than heading styles.
    """
    text = paragraph.text.strip()
    if not text:
        return False
    runs_with_text = [r for r in paragraph.runs if r.text.strip()]
    if not runs_with_text:
        return False
    return all(r.bold for r in runs_with_text)


def _classify_section(heading: str) -> Literal["article", "signoff", "meta"]:
    """Classify section type based on heading content."""
    heading_lower = heading.lower()
    if any(
        kw in heading_lower
        for kw in ("speaking date", "media report", "week in review", "this week")
    ):
        return "meta"
    if heading_lower in ("peter", "peter chapman"):
        return "signoff"
    return "article"


def _flush_section(
    heading: str,
    body_parts: list[str],
) -> NewsletterSection:
    """Build a NewsletterSection from accumulated heading and body parts."""
    body = "\n".join(body_parts)
    return NewsletterSection(
        heading=heading,
        body=body,
        section_type=_classify_section(heading),
    )


def _detect_channel(filename: str) -> str:
    """Return 'thegrower' for cross-posted Grower magazine variants, else 'skufood'."""
    return "thegrower" if "thegrower" in filename.lower() else "skufood"


def parse_newsletter(path: Path) -> Newsletter:
    """Parse a single .docx newsletter into a Newsletter object.

    Args:
        path: Path to the .docx file.

    Returns:
        Parsed Newsletter with sections split by bold headings.

    Raises:
        ValueError: If the blog number cannot be extracted from the filename.
    """
    blog_number = _extract_blog_number(path.stem)
    channel = _detect_channel(path.stem)
    doc = Document(str(path))

    paragraphs: list[Paragraph] = [p for p in doc.paragraphs if p.text.strip()]

    if not paragraphs:
        return Newsletter(
            blog_number=blog_number,
            title="",
            source_file=path.name,
            raw_word_count=0,
            channel=channel,  # type: ignore[arg-type]
        )

    title = ""
    sections: list[NewsletterSection] = []
    current_heading = ""
    current_body_parts: list[str] = []

    for para in paragraphs:
        text = para.text.strip()
        if _is_heading(para):
            if current_heading or current_body_parts:
                sections.append(_flush_section(current_heading, current_body_parts))
            if not title:
                title = text
            current_heading = text
            current_body_parts = []
        else:
            current_body_parts.append(text)

    if current_heading or current_body_parts:
        sections.append(_flush_section(current_heading, current_body_parts))

    if not title and paragraphs:
        title = paragraphs[0].text.strip()[:100]

    # raw_word_count covers all paragraph text (including pre-heading content
    # that may not appear in sections). ProcessedDocument.word_count is
    # recomputed from section text only — the two may differ slightly.
    full_text = "\n".join(p.text for p in paragraphs)
    word_count = len(full_text.split())

    return Newsletter(
        blog_number=blog_number,
        title=title,
        sections=tuple(sections),
        raw_word_count=word_count,
        source_file=path.name,
        channel=channel,  # type: ignore[arg-type]
    )


def parse_all_newsletters(
    directory: Path,
    *,
    show_progress: bool = False,
) -> list[Newsletter]:
    """Parse all .docx files in a directory into Newsletter objects.

    Files that fail to parse are logged and skipped.

    Args:
        directory: Path to the directory containing .docx files.
        show_progress: If True, display a tqdm progress bar.

    Returns:
        List of successfully parsed newsletters, sorted by blog number.
    """
    docx_files = sorted(directory.glob("*.docx"))
    if not docx_files:
        logger.warning("no_docx_files_found", directory=str(directory))
        return []

    iterable: Iterable[Path] = docx_files
    if show_progress:
        from tqdm import tqdm

        iterable = tqdm(docx_files, desc="Parsing newsletters")

    newsletters: list[Newsletter] = []
    errors: list[str] = []

    for path in iterable:
        try:
            newsletter = parse_newsletter(path)
            newsletters.append(newsletter)
        except (ValueError, OSError, zipfile.BadZipFile) as e:
            errors.append(f"{path.name}: {e}")
            logger.error("parse_failed", file=path.name, error=str(e))

    newsletters.sort(key=lambda n: n.blog_number)

    logger.info(
        "parse_complete",
        total_files=len(docx_files),
        parsed=len(newsletters),
        errors=len(errors),
    )

    if errors:
        logger.warning("parse_errors", errors=errors)

    return newsletters
