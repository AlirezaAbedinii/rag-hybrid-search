"""BM25 sparse index: build/query over the same chunk corpus; persist as pickle.

Kept in sync with the vector store: :func:`rag.indexing.index_path` upserts the
same chunks (by the same stable ``chunk_id``) into both, so the two report the
same count after ingest.

The tokenizer is the design point. Technical docs win on **exact tokens** —
error codes (``FERRY-429``), config keys (``ferry.worker.concurrency``) — so in
addition to plain alphanumeric words it emits compound tokens (words joined by
``.``/``-``/``_``) intact. A query for "FERRY-429" therefore matches the chunk
containing that literal code even when embedding similarity is weak.

Texts and metadata are persisted alongside the token corpus so sparse retrieval
can return full scored chunks without touching Chroma. ``rank_bm25`` is a tiny
pure-Python dependency; it is imported lazily all the same.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from ..ingestion.chunkers import Chunk

# Plain words plus compound technical tokens kept intact.
_WORD_RE = re.compile(r"[a-z0-9]+")
_COMPOUND_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus dotted/hyphenated compounds kept whole."""
    lower = text.lower()
    return _WORD_RE.findall(lower) + _COMPOUND_RE.findall(lower)


class BM25Index:
    """An upsertable, persistent BM25 index over chunk texts."""

    def __init__(self) -> None:
        # chunk_id -> (text, metadata); insertion order is preserved and
        # deterministic given the same upsert sequence.
        self._entries: dict[str, tuple[str, dict]] = {}
        self._bm25 = None  # rebuilt lazily after mutations
        self._ids: list[str] = []

    # -- building ----------------------------------------------------------
    def upsert(self, chunks: list[Chunk]) -> int:
        """Insert or replace chunks by ``chunk_id``; return how many."""
        for chunk in chunks:
            self._entries[chunk.chunk_id] = (chunk.text, chunk.metadata())
        self._bm25 = None
        return len(chunks)

    def count(self) -> int:
        return len(self._entries)

    def _ensure_built(self):
        if self._bm25 is None:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:  # pragma: no cover - only without the dep
                raise ImportError(
                    "BM25 requires 'rank-bm25'. Install: pip install -e \".[indexing]\""
                ) from exc
            self._ids = list(self._entries)
            corpus = [tokenize(self._entries[cid][0]) for cid in self._ids]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    # -- querying ----------------------------------------------------------
    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float, str, dict]]:
        """Top-k ``(chunk_id, score, text, metadata)`` for a query string.

        Scores are raw BM25 (unbounded, corpus-dependent); ranking is what
        matters downstream — RRF fuses by rank, not score.
        """
        if not self._entries:
            return []
        bm25 = self._ensure_built()
        scores = bm25.get_scores(tokenize(text))
        ranked = sorted(
            zip(self._ids, scores, strict=True), key=lambda pair: (-pair[1], pair[0])
        )[:top_k]
        return [
            (cid, float(score), self._entries[cid][0], self._entries[cid][1])
            for cid, score in ranked
        ]

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._entries, fh)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        """Load a persisted index; raises FileNotFoundError with a fix hint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Ingest first: python scripts/seed.py"
            )
        index = cls()
        with open(path, "rb") as fh:
            index._entries = pickle.load(fh)
        return index

    @classmethod
    def load_or_new(cls, path: str | Path) -> BM25Index:
        """Load if present, else an empty index (ingest-time convenience)."""
        try:
            return cls.load(path)
        except FileNotFoundError:
            return cls()
