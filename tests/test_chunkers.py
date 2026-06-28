"""Phase 1 tests for src/rag/ingestion/chunkers.py (deterministic, no network)."""
from __future__ import annotations

import pytest

from rag.config import Settings
from rag.ingestion.chunkers import (
    FixedSizeChunker,
    chunk_document,
    get_chunker,
)
from rag.ingestion.loaders import RawBlock, RawDocument


def test_fixed_chunker_respects_size_and_overlap() -> None:
    text = "".join(str(i % 10) for i in range(1000))  # 1000 chars
    chunker = FixedSizeChunker(chunk_size=400, overlap=100)
    windows = chunker.split_text(text)

    assert all(len(w) <= 400 for w in windows)
    # Step = size - overlap = 300; consecutive windows share `overlap` chars.
    for a, b in zip(windows, windows[1:], strict=False):
        assert a[-100:] == b[:100]
    # Reassembling by the step must reproduce the original text exactly.
    rebuilt = windows[0] + "".join(w[100:] for w in windows[1:])
    assert rebuilt == text


def test_fixed_chunker_short_text_single_window() -> None:
    chunker = FixedSizeChunker(chunk_size=400, overlap=100)
    assert chunker.split_text("short") == ["short"]
    assert chunker.split_text("") == []


def test_fixed_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, overlap=100)


def test_chunk_document_tags_metadata() -> None:
    doc = RawDocument(
        source_file="doc.md",
        doc_type="markdown",
        blocks=[
            RawBlock(text="alpha body", section_heading="Intro"),
            RawBlock(text="beta body", section_heading="Details"),
        ],
    )
    chunks = chunk_document(doc, FixedSizeChunker(chunk_size=400, overlap=50))
    assert len(chunks) == 2
    assert [c.ordinal for c in chunks] == [0, 1]
    assert all(c.strategy == "fixed" for c in chunks)
    assert all(c.source_file == "doc.md" for c in chunks)
    assert all(len(c.chunk_id) == 16 for c in chunks)
    assert chunks[0].section_heading == "Intro"
    # Metadata is Chroma-safe: no None values.
    meta = chunks[1].metadata()
    assert meta["section_heading"] == "Details"
    assert meta["page"] == -1  # None page -> sentinel
    assert None not in meta.values()


def test_chunk_ids_are_stable_and_unique() -> None:
    doc = RawDocument(
        source_file="doc.txt",
        doc_type="text",
        blocks=[RawBlock(text="a" * 1000)],
    )
    chunker = FixedSizeChunker(chunk_size=300, overlap=50)
    first = chunk_document(doc, chunker)
    second = chunk_document(doc, chunker)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]  # stable
    assert len({c.chunk_id for c in first}) == len(first)  # unique


def test_get_chunker_factory() -> None:
    fixed = get_chunker(Settings(_env_file=None, chunk_strategy="fixed", chunk_size=500))
    assert isinstance(fixed, FixedSizeChunker)
    assert fixed.chunk_size == 500
    with pytest.raises(NotImplementedError):
        get_chunker(Settings(_env_file=None, chunk_strategy="recursive"))
