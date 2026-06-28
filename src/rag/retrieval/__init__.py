"""Retrieval: dense, sparse, RRF fusion, cross-encoder rerank.

The public entrypoint is :func:`build_retriever`, which routes on ``mode``:

* ``dense``  — MVP: embed query -> Chroma top-k (cosine).
* ``hybrid`` — V1: dense + BM25 -> RRF fusion -> cross-encoder rerank.

Both retrievers expose the same ``retrieve(query, top_k, stopwatch) -> list[ScoredChunk]``
interface so callers (the pipeline, the API) are mode-agnostic.
"""
from __future__ import annotations

from ..config import Settings, get_settings
from ..indexing.vector_store import ScoredChunk
from .dense import DenseRetriever, SupportsEmbedQuery, SupportsQuery

VALID_MODES = ("dense", "hybrid")

__all__ = [
    "DenseRetriever",
    "ScoredChunk",
    "SupportsEmbedQuery",
    "SupportsQuery",
    "VALID_MODES",
    "build_retriever",
]


def build_retriever(
    mode: str | None = None,
    *,
    settings: Settings | None = None,
    embedder: SupportsEmbedQuery | None = None,
    store: SupportsQuery | None = None,
) -> DenseRetriever:
    """Build the retriever for ``mode`` (defaults to ``settings.default_mode``).

    ``embedder``/``store`` may be injected (tests, custom backends); otherwise the
    real provider clients are constructed from settings. ``hybrid`` raises
    :class:`NotImplementedError` until the V1 sparse/fusion/rerank stack lands.
    """
    settings = settings or get_settings()
    mode = mode or settings.default_mode
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {VALID_MODES}.")

    if mode == "hybrid":
        raise NotImplementedError(
            "Hybrid retrieval (BM25 + RRF + rerank) is a V1 feature; use mode='dense' for the MVP."
        )

    # mode == "dense"
    if embedder is not None and store is not None:
        return DenseRetriever(embedder=embedder, store=store, default_top_k=settings.top_k)
    return DenseRetriever.from_settings(settings)
