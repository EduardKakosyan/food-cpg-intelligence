"""Tests for data models."""

import pytest
from pydantic import ValidationError

from food_cpg_intelligence.data.models import (
    Newsletter,
    NewsletterSection,
    ProcessedDocument,
)


def test_newsletter_section_is_frozen() -> None:
    s = NewsletterSection(heading="Title", body="Content")
    assert s.heading == "Title"
    assert s.section_type == "article"
    with pytest.raises(ValidationError):
        s.heading = "Mutated"  # type: ignore[misc]


def test_newsletter_full_text() -> None:
    sections = (
        NewsletterSection(heading="Heading 1", body="Body 1"),
        NewsletterSection(heading="Heading 2", body="Body 2"),
    )
    n = Newsletter(blog_number=100, title="Test", sections=sections)
    assert "Heading 1" in n.full_text
    assert "Body 1" in n.full_text
    assert "Heading 2" in n.full_text


def test_newsletter_empty_sections() -> None:
    n = Newsletter(blog_number=1, title="Empty")
    assert n.full_text == ""
    assert n.raw_word_count == 0


def test_processed_document_from_newsletter() -> None:
    sections = (NewsletterSection(heading="Trends", body="Some content here about trends."),)
    n = Newsletter(
        blog_number=42,
        title="Test Newsletter",
        sections=sections,
        raw_word_count=50,
        source_file="SKUFoodblog42.docx",
    )
    doc = ProcessedDocument.from_newsletter(n)
    assert doc.doc_id == "newsletter-42"
    assert doc.content_type == "newsletter"
    assert doc.title == "Test Newsletter"
    assert "Trends" in doc.full_text
    assert doc.metadata["blog_number"] == 42
    assert doc.metadata["section_count"] == 1
