"""Tests for ingest-time deduplication (deterministic, no network)."""
from __future__ import annotations

from dataclasses import dataclass

from rag.ingestion.chunkers import Chunk
from rag.ingestion.dedup import filter_duplicates


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source_file="d.md", strategy="fixed", ordinal=0)


@dataclass
class Hit:
    chunk_id: str
    score: float


class FakeStore:
    """Returns a preset nearest hit per query vector's first component."""

    def __init__(self, hits: dict[float, Hit]) -> None:
        self._hits = hits

    def count(self) -> int:
        return len(self._hits)

    def query(self, query_embedding, top_k):
        hit = self._hits.get(query_embedding[0])
        return [hit] if hit else []


class EmptyStore:
    def count(self) -> int:
        return 0

    def query(self, query_embedding, top_k):  # pragma: no cover - never called
        raise AssertionError("empty store must not be queried")


def test_near_identical_to_store_is_skipped_distinct_kept() -> None:
    chunks = [_chunk("new-1", "dup of existing"), _chunk("new-2", "novel content")]
    vectors = [[0.1, 0.0], [0.2, 0.0]]
    store = FakeStore({0.1: Hit("existing-9", 0.99), 0.2: Hit("existing-9", 0.30)})

    kept, kept_vecs, skipped = filter_duplicates(chunks, vectors, store, threshold=0.95)

    assert [c.chunk_id for c in kept] == ["new-2"]
    assert kept_vecs == [[0.2, 0.0]]
    assert skipped == ["new-1"]


def test_same_chunk_id_match_is_upsert_not_duplicate() -> None:
    chunks = [_chunk("same-id", "re-ingested content")]
    store = FakeStore({0.5: Hit("same-id", 1.0)})  # perfect match with itself

    kept, _, skipped = filter_duplicates(chunks, [[0.5, 0.0]], store, threshold=0.95)

    assert [c.chunk_id for c in kept] == ["same-id"]
    assert skipped == []


def test_batch_internal_duplicate_is_skipped() -> None:
    chunks = [_chunk("a", "text"), _chunk("b", "same text repeated")]
    vectors = [[1.0, 0.0], [1.0, 0.000001]]  # nearly identical directions
    kept, _, skipped = filter_duplicates(chunks, vectors, EmptyStore(), threshold=0.95)

    assert [c.chunk_id for c in kept] == ["a"]
    assert skipped == ["b"]


def test_below_threshold_pairs_are_kept() -> None:
    chunks = [_chunk("a", "one"), _chunk("b", "two")]
    vectors = [[1.0, 0.0], [0.0, 1.0]]  # orthogonal
    kept, _, skipped = filter_duplicates(chunks, vectors, EmptyStore(), threshold=0.95)
    assert len(kept) == 2 and skipped == []
