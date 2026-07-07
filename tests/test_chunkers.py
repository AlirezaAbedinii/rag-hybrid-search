"""Phase 1 tests for src/rag/ingestion/chunkers.py (deterministic, no network)."""
from __future__ import annotations

import pytest

from rag.config import Settings
from rag.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
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


# --- recursive chunker ------------------------------------------------------
MD_TEXT = (
    "# Title\n\nIntro paragraph. " + "Filler sentence here. " * 20 + "\n\n"
    "## Section A\n\n" + "Alpha content sentence. " * 20 + "\n\n"
    "## Section B\n\n" + "Beta content sentence. " * 20
)


def test_recursive_splits_on_headers_and_respects_size() -> None:
    chunker = RecursiveChunker(chunk_size=400)
    pieces = chunker.split_text(MD_TEXT)
    assert all(len(p) <= 400 for p in pieces)
    # Header lines start their own pieces (structure-aware boundaries).
    starts = [p.splitlines()[0] for p in pieces if p.startswith("#")]
    assert any(s.startswith("# Title") for s in starts)
    assert any(s.startswith("## Section A") for s in starts)
    assert any(s.startswith("## Section B") for s in starts)


def test_recursive_short_text_is_one_piece_and_deterministic() -> None:
    chunker = RecursiveChunker(chunk_size=400)
    assert chunker.split_text("short text") == ["short text"]
    assert chunker.split_text(MD_TEXT) == chunker.split_text(MD_TEXT)


def test_recursive_hard_cuts_single_oversized_sentence() -> None:
    long_sentence = "x" * 950  # no separators at all
    pieces = RecursiveChunker(chunk_size=400).split_text(long_sentence)
    assert all(len(p) <= 400 for p in pieces)
    assert "".join(pieces) == long_sentence


def test_recursive_strategy_tag_on_chunks() -> None:
    doc = RawDocument("d.md", "markdown", [RawBlock(text=MD_TEXT)])
    chunks = chunk_document(doc, RecursiveChunker(chunk_size=400))
    assert all(c.strategy == "recursive" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# --- semantic chunker ---------------------------------------------------------
class TopicEmbedder:
    """Deterministic fake: same first word -> identical vector, else orthogonal."""

    _topics = {"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0], "gamma": [0.0, 0.0, 1.0]}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._topics[t.split()[0].lower()] for t in texts]


def test_semantic_breaks_on_topic_shift() -> None:
    text = (
        "Alpha one is about the first topic. Alpha two continues it. "
        "Beta one changes the subject entirely. Beta two continues that. "
        "Gamma one is a third topic."
    )
    chunker = SemanticChunker(embedder=TopicEmbedder(), chunk_size=120, breakpoint_threshold=0.5)
    pieces = chunker.split_text(text)
    # Three topic runs -> pieces built from runs; each stays within budget and
    # no piece mixes alpha with beta/gamma sentences at a run boundary.
    assert len(pieces) >= 2
    assert all(len(p) <= 120 for p in pieces)
    assert any("Alpha one" in p and "Alpha two" in p for p in pieces)  # same-topic kept together
    assert not any("Alpha two" in p and "Beta one" in p for p in pieces)  # break at shift


def test_semantic_short_text_skips_embedding() -> None:
    class Boom:
        def embed_texts(self, texts):
            raise AssertionError("must not embed when text fits the budget")

    chunker = SemanticChunker(embedder=Boom(), chunk_size=500)
    assert chunker.split_text("fits in one chunk.") == ["fits in one chunk."]


def test_get_chunker_factory() -> None:
    fixed = get_chunker(Settings(_env_file=None, chunk_strategy="fixed", chunk_size=500))
    assert isinstance(fixed, FixedSizeChunker)
    assert fixed.chunk_size == 500

    recursive = get_chunker(Settings(_env_file=None, chunk_strategy="recursive", chunk_size=300))
    assert isinstance(recursive, RecursiveChunker)
    assert recursive.chunk_size == 300

    semantic = get_chunker(
        Settings(_env_file=None, chunk_strategy="semantic", semantic_breakpoint_threshold=0.4),
        embedder=TopicEmbedder(),
    )
    assert isinstance(semantic, SemanticChunker)
    assert semantic.breakpoint_threshold == 0.4
