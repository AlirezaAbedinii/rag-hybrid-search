"""Deduplication: skip chunks whose cosine similarity to existing content exceeds a threshold.

Runs at index time, between embedding and storing (see
:func:`rag.indexing.index_path`). Two comparisons per incoming chunk:

* against the **store** — the nearest already-indexed chunk (skipped when the
  match is the chunk's own id: re-ingesting is an idempotent upsert, not a dupe);
* against **earlier chunks in the same batch** that were already accepted.

Threshold comes from config (``dedup_cosine_threshold``, default 0.95). Skips
are logged and counted so the ingest summary can report them.
"""
from __future__ import annotations

import logging
import math
from typing import Protocol

from .chunkers import Chunk

logger = logging.getLogger(__name__)

Vector = list[float]


class SupportsNearest(Protocol):
    """The slice of the vector store dedup needs (VectorStore satisfies it)."""

    def count(self) -> int: ...

    def query(self, query_embedding: Vector, top_k: int) -> list: ...  # ScoredChunk-like


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def filter_duplicates(
    chunks: list[Chunk],
    vectors: list[Vector],
    store: SupportsNearest,
    threshold: float,
) -> tuple[list[Chunk], list[Vector], list[str]]:
    """Return ``(kept_chunks, kept_vectors, skipped_chunk_ids)``.

    A chunk is skipped when its embedding is more similar than ``threshold`` to
    (a) a *different* chunk already in the store, or (b) a chunk accepted
    earlier in this batch.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must be the same length")

    kept: list[Chunk] = []
    kept_vectors: list[Vector] = []
    skipped: list[str] = []
    store_has_content = store.count() > 0

    for chunk, vector in zip(chunks, vectors, strict=True):
        duplicate_of: str | None = None

        if store_has_content:
            nearest = store.query(vector, top_k=1)
            if nearest:
                top = nearest[0]
                if top.chunk_id != chunk.chunk_id and top.score > threshold:
                    duplicate_of = top.chunk_id

        if duplicate_of is None:
            for accepted, accepted_vector in zip(kept, kept_vectors, strict=True):
                if _cosine(vector, accepted_vector) > threshold:
                    duplicate_of = accepted.chunk_id
                    break

        if duplicate_of is None:
            kept.append(chunk)
            kept_vectors.append(vector)
        else:
            skipped.append(chunk.chunk_id)
            logger.info(
                "dedup: skipping chunk %s from %s (cosine > %.2f vs %s)",
                chunk.chunk_id,
                chunk.source_file,
                threshold,
                duplicate_of,
            )

    return kept, kept_vectors, skipped
