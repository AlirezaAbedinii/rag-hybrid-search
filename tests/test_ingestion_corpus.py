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


# --- index_path: Chroma + BM25 stay in sync (fakes; no Chroma, no network) ----
class HashEmbedder:
    """Deterministic per-text vectors; distinct texts are dissimilar."""

    total_cost_usd = 0.0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([h[i] / 255.0 for i in range(16)])
        return out


class MemoryStore:
    """Minimal in-memory stand-in for VectorStore (add/count/query)."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[list[float], str, dict]] = {}

    def add(self, chunks, embeddings) -> int:
        for c, v in zip(chunks, embeddings, strict=True):
            self._rows[c.chunk_id] = (v, c.text, c.metadata())
        return len(chunks)

    def count(self) -> int:
        return len(self._rows)

    def query(self, query_embedding, top_k):
        import math

        from rag.indexing.vector_store import ScoredChunk

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        scored = [
            ScoredChunk(cid, text, cos(query_embedding, vec), meta)
            for cid, (vec, text, meta) in self._rows.items()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


def test_index_path_keeps_chroma_and_bm25_in_sync(settings: Settings, tmp_path) -> None:
    pytest.importorskip("rank_bm25")
    from rag.indexing import index_path
    from rag.indexing.bm25_index import BM25Index

    store = MemoryStore()
    bm25_path = tmp_path / "bm25.pkl"
    summary = index_path(
        settings.corpus_dir,
        settings=settings,
        embedder=HashEmbedder(),
        store=store,
        bm25_path=bm25_path,
    )

    # Acceptance: both indexes report the same chunk count after ingest.
    assert summary.total_chunks_in_store == summary.bm25_chunks
    assert store.count() == BM25Index.load(bm25_path).count()
    assert summary.chunks_indexed > 0
    assert "dedup" in summary.timings_ms and "store" in summary.timings_ms

    # Re-ingest is idempotent for both stores (upsert by stable chunk_id).
    again = index_path(
        settings.corpus_dir,
        settings=settings,
        embedder=HashEmbedder(),
        store=store,
        bm25_path=bm25_path,
    )
    assert again.total_chunks_in_store == summary.total_chunks_in_store
    assert again.bm25_chunks == summary.bm25_chunks


def test_index_path_dedups_repeated_content(settings: Settings, tmp_path) -> None:
    pytest.importorskip("rank_bm25")
    from rag.indexing import index_path

    # A corpus with a duplicated file: same text, different filename.
    src = (settings.corpus_dir / "01-overview.md").read_text(encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "original.md").write_text(src, encoding="utf-8")
    (corpus / "copy.md").write_text(src, encoding="utf-8")

    summary = index_path(
        corpus,
        settings=settings,
        embedder=HashEmbedder(),
        store=MemoryStore(),
        bm25_path=tmp_path / "bm25.pkl",
    )

    # The copied file's chunks embed identically -> skipped as duplicates.
    assert summary.chunks_skipped_duplicates > 0
    assert summary.total_chunks_in_store == summary.bm25_chunks
    assert summary.chunks_indexed + summary.chunks_skipped_duplicates == (
        summary.chunks_indexed * 2
    )
