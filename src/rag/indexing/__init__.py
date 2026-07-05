"""Indexing: embeddings, vector store (Chroma), BM25 index.

:func:`index_path` is the shared "make this path searchable" operation — load,
normalize, chunk, embed, and upsert into the vector store — used by
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
    timings_ms: dict[str, float] = field(default_factory=dict)


def index_path(
    path: str | Path,
    *,
    settings: Settings | None = None,
    embedder=None,
    store=None,
    persist_processed: bool = False,
) -> IndexSummary:
    """Load, chunk, embed, and store every supported file under ``path``."""
    from ..ingestion import build_chunks_for_dir, build_chunks_for_file

    settings = settings or get_settings()
    if embedder is None:
        from .embeddings import get_embedding_client

        embedder = get_embedding_client(settings)
    if store is None:
        from .vector_store import VectorStore

        store = VectorStore.from_settings(settings)

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
    with sw.time("store"):
        stored = store.add(chunks, vectors)

    return IndexSummary(
        files=len({c.source_file for c in chunks}),
        chunks_indexed=stored,
        total_chunks_in_store=store.count(),
        embedding_cost_usd=round(getattr(embedder, "total_cost_usd", 0.0) - cost_before, 6),
        timings_ms=sw.as_dict(),
    )
