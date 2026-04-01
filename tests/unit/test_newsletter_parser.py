"""Tests for newsletter parser."""

from pathlib import Path

import pytest

from food_cpg_intelligence.data.newsletter_parser import (
    _extract_blog_number,
    parse_all_newsletters,
    parse_newsletter,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_extract_blog_number_standard() -> None:
    assert _extract_blog_number("SKUFoodblog100") == 100


def test_extract_blog_number_lowercase() -> None:
    assert _extract_blog_number("SKUfoodblog42") == 42


def test_extract_blog_number_high() -> None:
    assert _extract_blog_number("SKUFoodblog333") == 333


def test_extract_blog_number_invalid() -> None:
    with pytest.raises(ValueError, match="Cannot extract blog number"):
        _extract_blog_number("random_file")


def test_parse_sample_newsletter() -> None:
    path = FIXTURES / "SKUFoodblog999.docx"
    newsletter = parse_newsletter(path)
    assert newsletter.blog_number == 999
    assert newsletter.title == "Canadian Retail Trends for 2024"
    assert len(newsletter.sections) >= 3
    assert newsletter.raw_word_count > 0
    assert newsletter.source_file == "SKUFoodblog999.docx"


def test_parse_sample_has_article_sections() -> None:
    path = FIXTURES / "SKUFoodblog999.docx"
    newsletter = parse_newsletter(path)
    article_sections = [s for s in newsletter.sections if s.section_type == "article"]
    assert len(article_sections) >= 2


def test_parse_empty_newsletter() -> None:
    path = FIXTURES / "SKUFoodblog001.docx"
    newsletter = parse_newsletter(path)
    assert newsletter.blog_number == 1
    assert newsletter.raw_word_count == 0
    assert len(newsletter.sections) == 0


def test_parse_no_bold_newsletter() -> None:
    path = FIXTURES / "SKUFoodblog000.docx"
    newsletter = parse_newsletter(path)
    assert newsletter.blog_number == 0
    # Should still have content, just no bold-based sections
    assert newsletter.title != ""


def test_parse_all_newsletters_fixture_dir() -> None:
    newsletters = parse_all_newsletters(FIXTURES)
    assert len(newsletters) == 3
    # Should be sorted by blog number
    assert newsletters[0].blog_number < newsletters[1].blog_number


def test_parse_all_newsletters_empty_dir(tmp_path: Path) -> None:
    newsletters = parse_all_newsletters(tmp_path)
    assert newsletters == []
