"""Tests for text cleaner."""

from food_cpg_intelligence.data.cleaner import clean_newsletter, clean_text
from food_cpg_intelligence.data.models import Newsletter, NewsletterSection


def test_clean_smart_quotes() -> None:
    assert clean_text("\u201cHello\u201d") == '"Hello"'
    assert clean_text("\u2018world\u2019") == "'world'"


def test_clean_em_dash() -> None:
    assert clean_text("food\u2014beverage") == "food--beverage"


def test_clean_en_dash() -> None:
    assert clean_text("2022\u20132024") == "2022-2024"


def test_clean_ellipsis() -> None:
    assert clean_text("wait\u2026") == "wait..."


def test_clean_nbsp() -> None:
    assert clean_text("non\xa0breaking") == "non breaking"


def test_clean_multi_whitespace() -> None:
    assert clean_text("too   many    spaces") == "too many spaces"


def test_clean_multi_newlines() -> None:
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_url_tracking_all_utm() -> None:
    result = clean_text("https://example.com/page?utm_source=email&utm_medium=link")
    assert "utm_source" not in result
    assert result == "https://example.com/page"


def test_clean_url_tracking_mixed_params() -> None:
    result = clean_text("https://example.com/page?utm_source=x&id=1&utm_campaign=y")
    assert "utm_source" not in result
    assert "utm_campaign" not in result
    assert result == "https://example.com/page?id=1"


def test_clean_url_tracking_no_utm() -> None:
    url = "https://example.com/page?id=1&name=test"
    assert clean_text(url) == url


def test_clean_url_tracking_utm_with_ampersand_prefix() -> None:
    result = clean_text("https://example.com?oly_enc_id=abc&utm_source=omeda&utm_medium=email")
    assert "utm_source" not in result
    assert "oly_enc_id=abc" in result


def test_clean_unicode_normalization() -> None:
    # NFKC normalizes compatibility characters
    assert clean_text("\ufb01") == "fi"  # fi ligature


def test_clean_strip_whitespace() -> None:
    assert clean_text("  hello  ") == "hello"


def test_clean_newsletter() -> None:
    n = Newsletter(
        blog_number=1,
        title="Test \u201cTitle\u201d",
        sections=(
            NewsletterSection(
                heading="Section\xa01",
                body="Body with\u2026 content",
            ),
        ),
        raw_word_count=10,
    )
    cleaned = clean_newsletter(n)
    assert cleaned.title == 'Test "Title"'
    assert cleaned.sections[0].heading == "Section 1"
    assert cleaned.sections[0].body == "Body with... content"
    # Preserved fields
    assert cleaned.blog_number == 1
    assert cleaned.raw_word_count == 10
