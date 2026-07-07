"""Indexing: embeddings, vector store (Chroma), BM25 index.

:func:`index_path` is the shared "make this path searchable" operation — load,
normalize, chunk, embed, **dedup**, and upsert into **both** stores (Chroma and
the BM25 sparse index, kept in sync by the same stable chunk ids) — used by
``scripts/ingest.py``, ``scripts/seed.py``, and ``POST /v1/ingest``. The
embedder/store are injectable so it is testable without providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from ..observability.metrics import Stopwatch


@dataclass
class IndexSummary:
    """What an :func:`index_path` run did, plus its cost/latency."""

    files: int
    chunks_indexed: int
    total_chunks_in_store: int
    embedding_cost_usd: float
    bm25_chunks: int = 0
    chunks_skipped_duplicates: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


def index_path(
    path: str | Path,
    *,
    settings: Settings | None = None,
    embedder=None,
    store=None,
    bm25_path: str | Path | None = None,
    persist_processed: bool = False,
) -> IndexSummary:
    """Load, chunk, embed, dedup, and store every supported file under ``path``.

    Chunks are upserted into Chroma and the persisted BM25 index in the same
    pass; near-duplicates (cosine > ``settings.dedup_cosine_threshold`` vs
    existing or earlier-in-batch content) are skipped from **both**, keeping
    the two stores at identical counts.
    """
    from ..ingestion import build_chunks_for_dir, build_chunks_for_file
    from ..ingestion.dedup import filter_duplicates
    from .bm25_index import BM25Index

    settings = settings or get_settings()
    if embedder is None:
        from .embeddings import get_embedding_client

        embedder = get_embedding_client(settings)
    if store is None:
        from .vector_store import VectorStore

        store = VectorStore.from_settings(settings)
    bm25_path = Path(bm25_path) if bm25_path else settings.bm25_index_path

    path = Path(path)
    sw = Stopwatch()
    with sw.time("load_chunk"):
        if path.is_dir():
            chunks = build_chunks_for_dir(path, settings=settings, persist=persist_processed)
        else:
            chunks = build_chunks_for_file(path, settings=settings, persist=persist_processed)

    cost_before = getattr(embedder, "total_cost_usd", 0.0)
    with sw.time("embed"):
        vectors = embedder.embed_texts([c.text for c in chunks])

    with sw.time("dedup"):
        kept, kept_vectors, skipped = filter_duplicates(
            chunks, vectors, store, settings.dedup_cosine_threshold
        )

    with sw.time("store"):
        stored = store.add(kept, kept_vectors)
        bm25 = BM25Index.load_or_new(bm25_path)
        bm25.upsert(kept)
        bm25.save(bm25_path)

    return IndexSummary(
        files=len({c.source_file for c in chunks}),
        chunks_indexed=stored,
        total_chunks_in_store=store.count(),
        embedding_cost_usd=round(getattr(embedder, "total_cost_usd", 0.0) - cost_before, 6),
        bm25_chunks=bm25.count(),
        chunks_skipped_duplicates=len(skipped),
        timings_ms=sw.as_dict(),
    )
