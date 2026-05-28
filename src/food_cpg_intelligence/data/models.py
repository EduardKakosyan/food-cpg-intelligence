"""Data models for the content ingestion pipeline."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class NewsletterSection(BaseModel, frozen=True):
    """A single section within a newsletter."""

    heading: str
    body: str
    section_type: Literal["article", "signoff", "meta"] = "article"


class Newsletter(BaseModel, frozen=True):
    """A parsed SKUFood newsletter."""

    blog_number: int
    title: str
    published_date: date | None = None
    author: str = "Peter Chapman"
    sections: tuple[NewsletterSection, ...] = ()
    raw_word_count: int = 0
    source_file: str = ""
    # Distribution channel: "skufood" (default) or "thegrower" (cross-posted version).
    # Used to disambiguate doc_id when the same blog number appears on both channels.
    channel: Literal["skufood", "thegrower"] = "skufood"

    @property
    def full_text(self) -> str:
        """Concatenate all section text into a single string."""
        return "\n\n".join(part for s in self.sections for part in (s.heading, s.body) if part)


ContentType = Literal["newsletter", "membership", "reference", "transcript"]


class ProcessedDocument(BaseModel, frozen=True):
    """Unified wrapper for any processed document in the corpus."""

    doc_id: str
    content_type: ContentType = "newsletter"
    title: str = ""
    full_text: str = ""
    word_count: int = 0
    source_file: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_newsletter(newsletter: Newsletter) -> ProcessedDocument:
        """Convert a Newsletter into a ProcessedDocument."""
        full_text = newsletter.full_text
        suffix = "" if newsletter.channel == "skufood" else f"-{newsletter.channel}"
        return ProcessedDocument(
            doc_id=f"newsletter-{newsletter.blog_number}{suffix}",
            content_type="newsletter",
            title=newsletter.title,
            full_text=full_text,
            word_count=len(full_text.split()),
            source_file=newsletter.source_file,
            metadata={
                "blog_number": newsletter.blog_number,
                "channel": newsletter.channel,
                "published_date": newsletter.published_date.isoformat()
                if newsletter.published_date
                else None,
                "author": newsletter.author,
                "section_count": len(newsletter.sections),
            },
        )
