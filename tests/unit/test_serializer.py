"""Tests for serializer (JSONL, Parquet, manifest)."""

import json
from pathlib import Path

import polars as pl

from food_cpg_intelligence.data.models import ProcessedDocument
from food_cpg_intelligence.data.serializer import (
    load_jsonl,
    save_jsonl,
    save_manifest,
    save_parquet,
    save_processed_documents,
)


def _make_docs() -> list[ProcessedDocument]:
    return [
        ProcessedDocument(
            doc_id="newsletter-1",
            content_type="newsletter",
            title="First Newsletter",
            full_text="Some text about food trends.",
            word_count=5,
            source_file="SKUFoodblog1.docx",
            metadata={"blog_number": 1, "author": "Peter"},
        ),
        ProcessedDocument(
            doc_id="newsletter-2",
            content_type="newsletter",
            title="Second Newsletter",
            full_text="More text about retail strategy.",
            word_count=5,
            source_file="SKUFoodblog2.docx",
            metadata={"blog_number": 2, "author": "Peter"},
        ),
    ]


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    docs = _make_docs()
    path = tmp_path / "test.jsonl"
    save_jsonl(docs, path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].doc_id == "newsletter-1"
    assert loaded[1].title == "Second Newsletter"


def test_parquet_save(tmp_path: Path) -> None:
    docs = _make_docs()
    path = tmp_path / "test.parquet"
    save_parquet(docs, path)
    df = pl.read_parquet(path)
    assert len(df) == 2
    assert "doc_id" in df.columns
    assert "meta_blog_number" in df.columns


def test_manifest_save(tmp_path: Path) -> None:
    docs = _make_docs()
    path = tmp_path / "manifest.json"
    save_manifest(docs, path)
    with path.open() as f:
        manifest = json.load(f)
    assert manifest["total_documents"] == 2
    assert manifest["total_words"] == 10
    assert manifest["avg_words_per_doc"] == 5.0
    assert manifest["by_content_type"]["newsletter"] == 2
    assert len(manifest["documents"]) == 2


def test_save_processed_documents_creates_all_files(tmp_path: Path) -> None:
    docs = _make_docs()
    save_processed_documents(docs, tmp_path)
    assert (tmp_path / "corpus.jsonl").exists()
    assert (tmp_path / "corpus.parquet").exists()
    assert (tmp_path / "corpus_manifest.json").exists()
