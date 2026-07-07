"""Sparse retrieval: BM25 top-k over the chunk corpus.

Loads the persisted BM25 index built at ingest time (see
``rag.indexing.index_path``) and returns :class:`ScoredChunk` results — the same
shape dense retrieval produces, so fusion treats both sources uniformly. BM25
scores are raw/unbounded; downstream RRF fuses by **rank**, so no normalization
is needed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Settings, get_settings
from ..indexing.vector_store import ScoredChunk
from ..observability.metrics import Stopwatch


class SupportsBM25Query(Protocol):
    """The slice of BM25Index the retriever needs (injectable in tests)."""

    def query(self, text: str, top_k: int) -> list[tuple[str, float, str, dict]]: ...


@dataclass
class SparseRetriever:
    """BM25 keyword retrieval over the persisted chunk corpus."""

    index: SupportsBM25Query
    default_top_k: int = 10

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> SparseRetriever:
        """Load the BM25 pickle; a missing index names the seed command."""
        from ..indexing.bm25_index import BM25Index

        settings = settings or get_settings()
        return cls(
            index=BM25Index.load(Path(settings.bm25_index_path)),
            default_top_k=settings.top_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        stopwatch: Stopwatch | None = None,
    ) -> list[ScoredChunk]:
        k = top_k or self.default_top_k
        sw = stopwatch or Stopwatch()
        with sw.time("sparse"):
            hits = self.index.query(query, top_k=k)
        return [
            ScoredChunk(chunk_id=cid, text=text, score=score, metadata=metadata)
            for cid, score, text, metadata in hits
        ]
