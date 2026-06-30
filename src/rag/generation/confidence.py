"""Confidence signals.

MVP: :func:`retrieval_confidence` — a simple, deterministic proxy derived from the
top retrieval score. It gates the "I don't know" path: when it falls below
``settings.retrieval_confidence_threshold`` the pipeline refuses instead of
generating.

V1: a *composite* confidence combining retrieval confidence, citation coverage,
and answer completeness (see :func:`composite_confidence`, not yet implemented).
"""
from __future__ import annotations

from typing import Protocol


class SupportsScore(Protocol):
    score: float


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def retrieval_confidence(results: list[SupportsScore]) -> float:
    """Proxy retrieval confidence in [0, 1] from the top result's score.

    Returns 0.0 on empty retrieval. Cosine similarity scores can sit slightly
    outside [0, 1], so the value is clamped.
    """
    if not results:
        return 0.0
    return _clamp01(max(r.score for r in results))


# --- V1 placeholder -------------------------------------------------------
def composite_confidence(*args, **kwargs) -> float:  # pragma: no cover
    """V1: f(retrieval confidence, citation coverage, answer completeness)."""
    raise NotImplementedError("Composite confidence score is a V1 feature.")
