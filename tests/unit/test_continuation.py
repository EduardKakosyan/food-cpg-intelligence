"""Tests for the continuation-mode SFT generator."""

from __future__ import annotations

import pytest

from food_cpg_intelligence.data.models import ProcessedDocument
from food_cpg_intelligence.training.continuation import (
    ChunkConfig,
    _approx_tokens,
    _greedy_chunk,
    _split_paragraphs,
    _topic_from_document,
    generate_from_corpus,
    generate_from_document,
)


def _doc(
    doc_id: str = "test-1",
    title: str = "How to do retail right",
    text: str = "Para one.\n\nPara two.\n\nPara three.",
    content_type: str = "newsletter",
) -> ProcessedDocument:
    return ProcessedDocument(
        doc_id=doc_id,
        content_type=content_type,  # type: ignore[arg-type]
        title=title,
        full_text=text,
        word_count=len(text.split()),
    )


@pytest.mark.unit
def test_split_paragraphs_drops_empty() -> None:
    assert _split_paragraphs("A\n\n\nB\n\n  \n\nC") == ["A", "B", "C"]


@pytest.mark.unit
def test_topic_strips_trailing_punctuation() -> None:
    doc = _doc(title="How to win? ")
    assert _topic_from_document(doc) == "how to win"


@pytest.mark.unit
def test_topic_falls_back_to_first_line_when_no_title() -> None:
    doc = _doc(title="", text="First line topic.\n\nBody body body.")
    assert _topic_from_document(doc) == "first line topic"


@pytest.mark.unit
def test_topic_returns_placeholder_for_empty_document() -> None:
    doc = _doc(title="", text="")
    assert _topic_from_document(doc) == "this topic"


@pytest.mark.unit
def test_greedy_chunk_groups_short_paragraphs() -> None:
    paras = ["word " * 80] * 5  # 80 words/para ~= 106 tokens/para
    cfg = ChunkConfig(target_tokens=300, min_tokens=100)
    chunks = _greedy_chunk(paras, cfg)
    # Each chunk should pack ~3 paragraphs before exceeding 300 tokens.
    assert len(chunks) >= 2
    for chunk in chunks:
        assert _approx_tokens(chunk) >= cfg.min_tokens


@pytest.mark.unit
def test_greedy_chunk_drops_standalone_chunk_below_minimum() -> None:
    # Each para individually exceeds target, so each starts its own chunk.
    # First chunk passes min; second chunk (the small tail) is dropped.
    paras = ["word " * 300, "small tail"]  # 300 words ~ 399 tokens >> target
    cfg = ChunkConfig(target_tokens=200, min_tokens=80)
    chunks = _greedy_chunk(paras, cfg)
    assert len(chunks) == 1
    assert "small tail" not in chunks[0]


@pytest.mark.unit
def test_greedy_chunk_merges_small_tail_into_previous_chunk() -> None:
    # A small tail that fits within target rolls into the previous chunk
    # rather than being dropped — by design.
    paras = ["word " * 100, "small tail"]
    cfg = ChunkConfig(target_tokens=300, min_tokens=80)
    chunks = _greedy_chunk(paras, cfg)
    assert len(chunks) == 1
    assert "small tail" in chunks[0]


@pytest.mark.unit
def test_generate_emits_triples_for_long_doc() -> None:
    body = "\n\n".join(["word " * 200 for _ in range(4)])  # 800 words ~ 1064 tokens
    doc = _doc(title="Margins matter", text=body)
    triples = generate_from_document(doc, config=ChunkConfig(target_tokens=400, min_tokens=200))
    assert len(triples) >= 1
    t = triples[0]
    assert t.source_doc_id == doc.doc_id
    assert t.triple_id.startswith("continuation-test-1-")
    assert "margins matter" in t.instruction.lower()
    assert t.question_type == "advice"
    assert t.category == "newsletter"


@pytest.mark.unit
def test_generate_returns_empty_for_empty_doc() -> None:
    doc = _doc(text="")
    assert generate_from_document(doc) == []


@pytest.mark.unit
def test_generate_scrubs_entities_in_response() -> None:
    body = (
        "Loblaws expects a 30% margin on premium SKUs. "
        "Their listing fees are $5,000 per item. " * 50  # repeat to exceed min_tokens
    )
    doc = _doc(title="Listing fees", text=body)
    triples = generate_from_document(doc, config=ChunkConfig(target_tokens=400, min_tokens=200))
    assert triples
    response = triples[0].response
    assert "Loblaws" not in response
    assert "$5,000" not in response
    assert "30%" not in response
    assert "[retailer]" in response


@pytest.mark.unit
def test_eliciting_prompts_vary_across_chunks() -> None:
    # Long enough to produce many chunks
    body = "\n\n".join(["word " * 200 for _ in range(20)])
    doc = _doc(title="Variety test", text=body)
    triples = generate_from_document(doc, config=ChunkConfig(target_tokens=300, min_tokens=150))
    instructions = {t.instruction for t in triples}
    # We should hit multiple templates given enough chunks
    assert len(instructions) >= 2


@pytest.mark.unit
def test_corpus_run_is_deterministic() -> None:
    body = "\n\n".join(["word " * 200 for _ in range(6)])
    docs = [_doc(doc_id=f"d-{i}", title=f"Topic {i}", text=body) for i in range(3)]
    first = generate_from_corpus(docs, config=ChunkConfig(target_tokens=400, min_tokens=200))
    second = generate_from_corpus(docs, config=ChunkConfig(target_tokens=400, min_tokens=200))
    assert [t.instruction for t in first] == [t.instruction for t in second]
    assert [t.response for t in first] == [t.response for t in second]
