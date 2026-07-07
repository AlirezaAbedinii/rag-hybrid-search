"""Retrieval: dense, sparse, RRF fusion, cross-encoder rerank.

The public entrypoint is :func:`build_retriever`, which routes on ``mode``:

* ``dense``  — embed query -> Chroma top-k (cosine).
* ``hybrid`` — dense + BM25 in parallel -> Reciprocal Rank Fusion (configurable
  weights) -> local cross-encoder rerank (top-20 -> top-5 by default).

Both retrievers expose the same
``retrieve(query, top_k, stopwatch) -> list[ScoredChunk]`` interface so callers
(the pipeline, the API) are mode-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..indexing.vector_store import ScoredChunk
from ..observability.metrics import Stopwatch
from .dense import DenseRetriever, SupportsEmbedQuery, SupportsQuery
from .fusion import rrf_fuse
from .rerank import Reranker, SupportsScorePairs
from .sparse import SparseRetriever

VALID_MODES = ("dense", "hybrid")

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "Reranker",
    "ScoredChunk",
    "SparseRetriever",
    "SupportsEmbedQuery",
    "SupportsQuery",
    "VALID_MODES",
    "build_retriever",
]


@dataclass
class HybridRetriever:
    """Dense + sparse -> RRF fusion -> cross-encoder rerank -> top-k.

    Both sources are asked for ``candidates`` results; RRF merges them by rank
    with the configured dense/sparse weights; the reranker keeps the best
    ``top_k``. Per-stage latency lands in the stopwatch as ``embed``/``dense``
    (inside the dense retriever), ``sparse``, ``fusion``, and ``rerank``.
    """

    dense: DenseRetriever
    sparse: SparseRetriever
    reranker: Reranker
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    candidates: int = 20
    default_top_k: int = 5
    mode: str = "hybrid"

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        stopwatch: Stopwatch | None = None,
    ) -> list[ScoredChunk]:
        k = top_k or self.default_top_k
        sw = stopwatch or Stopwatch()

        dense_hits = self.dense.retrieve(query, top_k=self.candidates, stopwatch=sw)
        sparse_hits = self.sparse.retrieve(query, top_k=self.candidates, stopwatch=sw)

        with sw.time("fusion"):
            fused = rrf_fuse(
                [[c.chunk_id for c in dense_hits], [c.chunk_id for c in sparse_hits]],
                weights=[self.dense_weight, self.sparse_weight],
                k=self.rrf_k,
            )
            # Resolve fused ids back to chunks (dense text wins on overlap).
            by_id = {c.chunk_id: c for c in sparse_hits}
            by_id.update({c.chunk_id: c for c in dense_hits})
            candidates = [by_id[cid] for cid, _ in fused[: self.candidates]]

        with sw.time("rerank"):
            return self.reranker.rerank(query, candidates, top_k=k)


def build_retriever(
    mode: str | None = None,
    *,
    settings: Settings | None = None,
    embedder: SupportsEmbedQuery | None = None,
    store: SupportsQuery | None = None,
    sparse_index=None,
    scorer: SupportsScorePairs | None = None,
) -> DenseRetriever | HybridRetriever:
    """Build the retriever for ``mode`` (defaults to ``settings.default_mode``).

    Dependencies (``embedder``/``store``/``sparse_index``/``scorer``) may be
    injected for tests or custom backends; otherwise the real clients are
    constructed from settings — which for hybrid requires a seeded BM25 index
    and the ``rerank`` extra.
    """
    settings = settings or get_settings()
    mode = mode or settings.default_mode
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {VALID_MODES}.")

    if embedder is not None and store is not None:
        dense = DenseRetriever(embedder=embedder, store=store, default_top_k=settings.top_k)
    else:
        dense = DenseRetriever.from_settings(settings)

    if mode == "dense":
        return dense

    # mode == "hybrid"
    if sparse_index is not None:
        sparse = SparseRetriever(index=sparse_index, default_top_k=settings.top_k)
    else:
        sparse = SparseRetriever.from_settings(settings)

    if scorer is None:
        from .rerank import CrossEncoderScorer

        scorer = CrossEncoderScorer(settings.reranker_model)

    return HybridRetriever(
        dense=dense,
        sparse=sparse,
        reranker=Reranker(scorer=scorer, top_k=settings.rerank_top_k),
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
        rrf_k=settings.rrf_k,
        candidates=settings.rerank_candidates,
        default_top_k=settings.rerank_top_k,
    )
