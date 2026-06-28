"""Dense retrieval (MVP): embed query -> Chroma top-k (k=10) by cosine.

:class:`DenseRetriever` depends only on small protocols — something that can
``embed_query`` and something that can ``query`` a vector store — so it is fully
testable with in-memory fakes (no Chroma, no network). Per-stage latency is
recorded into an optional :class:`Stopwatch` (``embed`` then ``dense``) to keep
the latency/cost story instrumented from day one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Settings, get_settings
from ..indexing.vector_store import ScoredChunk
from ..observability.metrics import Stopwatch

Vector = list[float]


class SupportsEmbedQuery(Protocol):
    """Anything that can turn a query string into a dense vector."""

    def embed_query(self, text: str) -> Vector: ...


class SupportsQuery(Protocol):
    """Anything that can return the top-k nearest chunks for a vector."""

    def query(self, query_embedding: Vector, top_k: int) -> list[ScoredChunk]: ...


@dataclass
class DenseRetriever:
    """Embed a query and fetch its nearest chunks from a vector store."""

    embedder: SupportsEmbedQuery
    store: SupportsQuery
    default_top_k: int = 10
    mode: str = "dense"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> DenseRetriever:
        """Build a retriever backed by the configured embedder + Chroma store.

        Constructs the real provider clients (needs the API key + Chroma extra);
        tests inject fakes via the normal constructor instead.
        """
        settings = settings or get_settings()
        from ..indexing.embeddings import get_embedding_client
        from ..indexing.vector_store import VectorStore

        return cls(
            embedder=get_embedding_client(settings),
            store=VectorStore.from_settings(settings),
            default_top_k=settings.top_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        stopwatch: Stopwatch | None = None,
    ) -> list[ScoredChunk]:
        """Return the top-k chunks for ``query``, ranked by similarity."""
        k = top_k or self.default_top_k
        sw = stopwatch or Stopwatch()
        with sw.time("embed"):
            vector = self.embedder.embed_query(query)
        with sw.time("dense"):
            results = self.store.query(vector, k)
        return results
