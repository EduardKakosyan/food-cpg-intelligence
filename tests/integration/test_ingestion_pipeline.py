"""Integration test: parse -> clean -> serialize roundtrip."""

from pathlib import Path

from food_cpg_intelligence.data.cleaner import clean_newsletter
from food_cpg_intelligence.data.models import ProcessedDocument
from food_cpg_intelligence.data.newsletter_parser import parse_all_newsletters
from food_cpg_intelligence.data.serializer import load_jsonl, save_processed_documents

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_full_pipeline(tmp_path: Path) -> None:
    """End-to-end: fixture .docx -> parsed -> cleaned -> serialized -> loaded."""
    # Parse
    newsletters = parse_all_newsletters(FIXTURES)
    assert len(newsletters) == 3

    # Clean
    cleaned = [clean_newsletter(n) for n in newsletters]

    # Convert
    docs = [ProcessedDocument.from_newsletter(n) for n in cleaned]
    assert len(docs) == 3

    # Serialize
    save_processed_documents(docs, tmp_path)

    # Load back
    loaded = load_jsonl(tmp_path / "corpus.jsonl")
    assert len(loaded) == 3

    # Verify content preserved
    ids = {d.doc_id for d in loaded}
    assert "newsletter-999" in ids
    assert "newsletter-0" in ids
    assert "newsletter-1" in ids

    # Verify word counts are positive for non-empty docs
    for doc in loaded:
        if doc.doc_id != "newsletter-1":  # blog 1 is the empty fixture
            assert doc.word_count > 0
