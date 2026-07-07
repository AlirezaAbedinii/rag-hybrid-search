"""Cross-encoder reranker: score top-N candidates -> keep top-k.

A cross-encoder reads the *(query, chunk)* pair jointly, so it is far more
precise than embedding similarity — and far slower, which is why it only
reranks the short fused candidate list (default top-20 -> top-5) instead of
searching the corpus.

The model (``ms-marco-MiniLM-L-6-v2``) runs locally via ``sentence-transformers``
(the ``rerank`` extra) — no LLM cost. Raw logits are squashed through a sigmoid
so downstream confidence math stays in [0, 1]. The scorer sits behind a
one-method protocol, so tests inject a deterministic fake and never load torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ..indexing.vector_store import ScoredChunk


class SupportsScorePairs(Protocol):
    """Scores (query, text) pairs; higher = more relevant."""

    def score_pairs(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """Local sentence-transformers cross-encoder (lazy import, cached model)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "Reranking requires 'sentence-transformers'. "
                    'Install: pip install -e ".[rerank]"'
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        model = self._ensure_model()
        return [float(s) for s in model.predict([(query, t) for t in texts])]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Reranker:
    """Rerank candidate chunks by cross-encoder relevance."""

    scorer: SupportsScorePairs
    top_k: int = 5

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        """Return the best ``top_k`` candidates re-scored and re-ordered.

        Result scores are sigmoid-normalized cross-encoder relevances (the
        original retrieval scores served their purpose upstream). Ties break by
        chunk_id for determinism.
        """
        if not candidates:
            return []
        raw = self.scorer.score_pairs(query, [c.text for c in candidates])
        rescored = [
            ScoredChunk(
                chunk_id=c.chunk_id, text=c.text, score=_sigmoid(s), metadata=c.metadata
            )
            for c, s in zip(candidates, raw, strict=True)
        ]
        rescored.sort(key=lambda c: (-c.score, c.chunk_id))
        return rescored[: top_k or self.top_k]
