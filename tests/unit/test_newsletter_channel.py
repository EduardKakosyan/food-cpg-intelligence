"""Tests for newsletter channel detection (skufood vs thegrower cross-posts)."""

from food_cpg_intelligence.data.models import Newsletter, ProcessedDocument
from food_cpg_intelligence.data.newsletter_parser import _detect_channel


def test_detect_channel_skufood_default() -> None:
    assert _detect_channel("SKUFoodblog140") == "skufood"
    assert _detect_channel("SKUfoodblog42") == "skufood"


def test_detect_channel_thegrower_variants() -> None:
    assert _detect_channel("SKUFood140thegrower") == "thegrower"
    assert _detect_channel("gpsnewsletter135thegrower") == "thegrower"
    assert _detect_channel("GPSNewsletter58thegrower") == "thegrower"


def test_processed_document_id_includes_channel_suffix() -> None:
    skufood = Newsletter(blog_number=140, title="t", source_file="SKUFoodblog140.docx")
    grower = Newsletter(
        blog_number=140,
        title="t",
        source_file="SKUFood140thegrower.docx",
        channel="thegrower",
    )
    assert ProcessedDocument.from_newsletter(skufood).doc_id == "newsletter-140"
    assert ProcessedDocument.from_newsletter(grower).doc_id == "newsletter-140-thegrower"
