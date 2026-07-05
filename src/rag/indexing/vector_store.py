"""ChromaDB wrapper: persist chunk embeddings + metadata, dense top-k query.

A thin, persistent wrapper over a single Chroma collection configured for cosine
distance. Chunks are upserted by their stable ``chunk_id`` so re-ingesting the
same corpus is idempotent. ``chromadb`` is imported lazily (it's heavy) so this
module imports without the ``indexing`` extra installed.

Phase 1 needs ``add`` + ``count``; ``query`` is provided here and built on by the
Phase 2 dense retriever.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..ingestion.chunkers import Chunk

Vector = list[float]


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with its similarity score (1 - cosine distance)."""

    chunk_id: str
    text: str
    score: float
    metadata: dict


class VectorStore:
    """Persistent Chroma collection of chunk embeddings + metadata."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "The vector store requires 'chromadb'. Install: pip install -e \".[indexing]\""
            ) from exc
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> VectorStore:
        settings = settings or get_settings()
        return cls(str(settings.chroma_persist_dir), settings.chroma_collection)

    def add(self, chunks: list[Chunk], embeddings: list[Vector]) -> int:
        """Upsert chunks + their embeddings by ``chunk_id``; return how many."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def list_sources(self) -> dict[str, int]:
        """Map of ``source_file`` -> chunk count across the whole collection."""
        res = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in res["metadatas"] or []:
            source = str(meta.get("source_file", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def query(self, query_embedding: Vector, top_k: int = 10) -> list[ScoredChunk]:
        """Return the ``top_k`` nearest chunks by cosine similarity."""
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            ScoredChunk(chunk_id=i, text=d, score=1.0 - float(dist), metadata=m)
            for i, d, m, dist in zip(ids, docs, metas, dists, strict=False)
        ]
