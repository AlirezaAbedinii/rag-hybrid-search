"""Phase 1 tests for src/rag/ingestion/loaders.py (deterministic, no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.loaders import (
    UnsupportedFormatError,
    is_supported,
    load_markdown,
    load_path,
    load_text,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_text_single_block() -> None:
    doc = load_text(FIXTURES / "sample.txt")
    assert doc.doc_type == "text"
    assert doc.source_file == "sample.txt"
    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert "line one" in block.text
    assert block.section_heading is None
    assert block.page is None


def test_load_markdown_splits_on_headings() -> None:
    doc = load_markdown(FIXTURES / "sample.md")
    assert doc.doc_type == "markdown"
    headings = {b.section_heading for b in doc.blocks}
    assert headings == {"Sample Title", "First Section", "Second Section"}


def test_load_markdown_preamble_inherits_title() -> None:
    doc = load_markdown(FIXTURES / "sample.md")
    by_heading = {b.section_heading: b.text for b in doc.blocks}
    assert "Intro paragraph" in by_heading["Sample Title"]


def test_load_markdown_fenced_hash_is_not_a_heading() -> None:
    doc = load_markdown(FIXTURES / "sample.md")
    # The '# this hash ...' line lives inside a code fence in First Section.
    assert all("this hash" not in (b.section_heading or "") for b in doc.blocks)
    first = next(b for b in doc.blocks if b.section_heading == "First Section")
    assert "ferry.worker.concurrency" in first.text


def test_load_pdf_one_block_per_page() -> None:
    pytest.importorskip("pypdf")
    doc = load_path(FIXTURES / "sample.pdf")
    assert doc.doc_type == "pdf"
    assert [b.page for b in doc.blocks] == [1, 2]
    assert "page one" in doc.blocks[0].text
    assert "FERRY-429" in doc.blocks[1].text


def test_load_path_dispatches_by_extension() -> None:
    assert load_path(FIXTURES / "sample.md").doc_type == "markdown"
    assert load_path(FIXTURES / "sample.txt").doc_type == "text"


def test_unsupported_format_raises(tmp_path: Path) -> None:
    html = tmp_path / "page.html"
    html.write_text("<h1>hi</h1>", encoding="utf-8")
    assert not is_supported(html)  # HTML is a V1 loader
    with pytest.raises(UnsupportedFormatError):
        load_path(html)
