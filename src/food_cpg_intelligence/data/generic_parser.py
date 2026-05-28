"""Format-agnostic parsers for non-newsletter Peter Chapman content.

Handles the Membership Content and Misc Content corpora: bold-heading docx,
xlsx workbooks, pdf reference docs, and pptx decks. Each parser returns a
`ProcessedDocument` directly so the caller can mix sources freely.

Newsletter parsing stays in `newsletter_parser.py` because it has the
blog-number convention and the article/signoff/meta section heuristic; this
module deliberately keeps the schema simpler — title, full_text, metadata —
and leaves topic-aware structure to downstream chunking.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from pypdf import PdfReader

from food_cpg_intelligence.data.cleaner import clean_text
from food_cpg_intelligence.data.models import ContentType, ProcessedDocument

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = structlog.stdlib.get_logger(__name__)

_SUPPORTED_DOCX = {".docx"}
_SUPPORTED_XLSX = {".xlsx"}
_SUPPORTED_PDF = {".pdf"}
_SUPPORTED_PPTX = {".pptx"}
_ALL_SUPPORTED = _SUPPORTED_DOCX | _SUPPORTED_XLSX | _SUPPORTED_PDF | _SUPPORTED_PPTX


def _doc_id_from_path(path: Path, prefix: str) -> str:
    """Stable doc_id from filename stem — prefixed to avoid collisions across sources."""
    stem = path.stem.replace(" ", "_")
    return f"{prefix}-{stem}"


def parse_docx(path: Path, *, content_type: ContentType, prefix: str) -> ProcessedDocument:
    """Parse a generic .docx into a ProcessedDocument.

    Uses the same bold-heading-is-section convention as newsletters but does
    not require a blog number; the first non-empty paragraph becomes the title.
    """
    doc = Document(str(path))
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return ProcessedDocument(
            doc_id=_doc_id_from_path(path, prefix),
            content_type=content_type,
            title="",
            full_text="",
            word_count=0,
            source_file=path.name,
            metadata={"format": "docx"},
        )

    title = clean_text(paragraphs[0].text.strip()[:200])
    full_text = clean_text("\n\n".join(p.text for p in paragraphs))

    return ProcessedDocument(
        doc_id=_doc_id_from_path(path, prefix),
        content_type=content_type,
        title=title,
        full_text=full_text,
        word_count=len(full_text.split()),
        source_file=path.name,
        metadata={"format": "docx", "paragraph_count": len(paragraphs)},
    )


def parse_xlsx(path: Path, *, content_type: ContentType, prefix: str) -> ProcessedDocument:
    """Flatten an .xlsx workbook into one document.

    Each sheet becomes a section; each non-empty row a paragraph with cells
    joined by tabs. xlsx files in this corpus are templates (promo plans,
    category reviews) so we keep them as flat reference text — not Q&A data.
    """
    workbook = load_workbook(str(path), data_only=True, read_only=True)
    sections: list[str] = []
    total_rows = 0
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        rows: list[str] = []
        for raw_row in sheet.iter_rows(values_only=True):
            cells = [str(c).strip() for c in raw_row if c is not None and str(c).strip()]
            if cells:
                # `|` survives clean_text's whitespace collapsing; tabs do not.
                rows.append(" | ".join(cells))
        if rows:
            sections.append(f"## {sheet_name}\n" + "\n".join(rows))
            total_rows += len(rows)
    workbook.close()

    full_text = clean_text("\n\n".join(sections))
    return ProcessedDocument(
        doc_id=_doc_id_from_path(path, prefix),
        content_type=content_type,
        title=path.stem,
        full_text=full_text,
        word_count=len(full_text.split()),
        source_file=path.name,
        metadata={"format": "xlsx", "sheet_count": len(sections), "row_count": total_rows},
    )


def parse_pdf(path: Path, *, content_type: ContentType, prefix: str) -> ProcessedDocument:
    """Extract text from a PDF, one page per paragraph block."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)

    full_text = clean_text("\n\n".join(pages))
    title = path.stem
    return ProcessedDocument(
        doc_id=_doc_id_from_path(path, prefix),
        content_type=content_type,
        title=title,
        full_text=full_text,
        word_count=len(full_text.split()),
        source_file=path.name,
        metadata={"format": "pdf", "page_count": len(reader.pages)},
    )


def parse_pptx(path: Path, *, content_type: ContentType, prefix: str) -> ProcessedDocument:
    """Extract slide text from a .pptx; each slide becomes a paragraph block."""
    presentation = Presentation(str(path))
    title = ""
    slides: list[str] = []
    for slide in presentation.slides:
        parts: list[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                text = "".join(run.text for run in paragraph.runs).strip()
                if text:
                    parts.append(text)
        if parts:
            slides.append("\n".join(parts))
            if not title:
                title = parts[0][:200]

    full_text = clean_text("\n\n".join(slides))
    return ProcessedDocument(
        doc_id=_doc_id_from_path(path, prefix),
        content_type=content_type,
        title=title or path.stem,
        full_text=full_text,
        word_count=len(full_text.split()),
        source_file=path.name,
        metadata={"format": "pptx", "slide_count": len(slides)},
    )


def parse_any(path: Path, *, content_type: ContentType, prefix: str) -> ProcessedDocument | None:
    """Dispatch to the right parser based on file extension; return None for unsupported."""
    suffix = path.suffix.lower()
    if suffix in _SUPPORTED_DOCX:
        return parse_docx(path, content_type=content_type, prefix=prefix)
    if suffix in _SUPPORTED_XLSX:
        return parse_xlsx(path, content_type=content_type, prefix=prefix)
    if suffix in _SUPPORTED_PDF:
        return parse_pdf(path, content_type=content_type, prefix=prefix)
    if suffix in _SUPPORTED_PPTX:
        return parse_pptx(path, content_type=content_type, prefix=prefix)
    return None


def parse_directory(
    directory: Path,
    *,
    content_type: ContentType,
    prefix: str,
    show_progress: bool = False,
) -> list[ProcessedDocument]:
    """Parse every supported file in `directory` into ProcessedDocuments.

    Files whose extension isn't in {docx,xlsx,pdf,pptx} are skipped silently.
    Files that error during parsing are logged and skipped — the run continues.
    """
    candidates = sorted(p for p in directory.iterdir() if p.suffix.lower() in _ALL_SUPPORTED)
    if not candidates:
        logger.warning("no_supported_files_found", directory=str(directory))
        return []

    iterable: Iterable[Path] = candidates
    if show_progress:
        from tqdm import tqdm

        iterable = tqdm(candidates, desc=f"Parsing {prefix}")

    docs: list[ProcessedDocument] = []
    errors: list[str] = []
    for path in iterable:
        try:
            doc = parse_any(path, content_type=content_type, prefix=prefix)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            logger.error("parse_failed", file=path.name, error=str(exc))
            continue
        if doc is not None and doc.word_count > 0:
            docs.append(doc)

    logger.info(
        "parse_directory_complete",
        directory=str(directory),
        candidates=len(candidates),
        parsed=len(docs),
        errors=len(errors),
    )
    if errors:
        logger.warning("parse_errors", errors=errors)
    return docs
