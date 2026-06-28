"""Phase 1 acceptance: chunk the real sample corpus deterministically.

Validates the Phase 1 acceptance criteria without any network call:
ingesting ``data/raw/ferry_docs`` yields N>0 chunks, each carrying complete
metadata (source_file, strategy, chunk_id; section_heading for markdown), and the
result is deterministic across runs.
"""
from __future__ import annotations

import pytest

from rag.config import Settings
from rag.ingestion import build_chunks_for_dir


@pytest.fixture
def settings() -> Settings:
    # Default config; corpus_dir resolves to <repo>/data/raw/ferry_docs.
    return Settings(_env_file=None)


def test_corpus_present(settings: Settings) -> None:
    assert settings.corpus_dir.is_dir(), f"missing corpus at {settings.corpus_dir}"
    assert list(settings.corpus_dir.glob("*.md")), "expected markdown docs in the corpus"


def test_corpus_chunks_have_complete_metadata(settings: Settings) -> None:
    chunks = build_chunks_for_dir(settings.corpus_dir, settings=settings)
    assert len(chunks) > 0

    sources = {c.source_file for c in chunks}
    assert len(sources) >= 6  # the ferry corpus has several markdown docs

    for c in chunks:
        assert c.source_file
        assert c.strategy == "fixed"
        assert len(c.chunk_id) == 16
        assert isinstance(c.ordinal, int)
        assert c.text.strip()
        # Markdown chunks must carry their section heading.
        assert c.section_heading, f"missing heading in {c.source_file} ordinal {c.ordinal}"
        # Chroma-safe metadata: never None.
        assert None not in c.metadata().values()


def test_corpus_chunking_is_deterministic(settings: Settings) -> None:
    first = build_chunks_for_dir(settings.corpus_dir, settings=settings)
    second = build_chunks_for_dir(settings.corpus_dir, settings=settings)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
