"""Confidence signals.

Two layers:

* :func:`retrieval_confidence` — deterministic proxy from the top retrieval
  score. Gates the "I don't know" path (below
  ``settings.retrieval_confidence_threshold`` the pipeline refuses before
  generating).
* :func:`composite_confidence` — the score returned with every answer: a
  weighted combination of retrieval confidence, **citation coverage** (share of
  verified citations judged supported), and **completeness** (share of answer
  sentences carrying at least one citation). Weights come from config. When a
  component is unavailable (e.g. verification disabled -> coverage is None) its
  weight is renormalized over the rest rather than silently counting as zero.

Monotonic in every input, clamped to [0, 1], deterministic given its inputs.
"""
from __future__ import annotations

import re
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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HAS_CITATION = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def answer_completeness(answer: str) -> float:
    """Share of answer sentences that carry at least one ``[n]`` citation."""
    sentences = [s for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _HAS_CITATION.search(s))
    return cited / len(sentences)


def composite_confidence(
    retrieval: float,
    coverage: float | None,
    completeness: float | None,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Weighted mix of the three signals, renormalized over available ones.

    ``coverage``/``completeness`` may be None (not measured); their weight is
    then redistributed proportionally instead of dragging the score to zero.
    """
    parts: list[tuple[float, float]] = [(_clamp01(retrieval), weights[0])]
    if coverage is not None:
        parts.append((_clamp01(coverage), weights[1]))
    if completeness is not None:
        parts.append((_clamp01(completeness), weights[2]))
    total_weight = sum(w for _, w in parts)
    if total_weight == 0:
        return 0.0
    return _clamp01(sum(v * w for v, w in parts) / total_weight)
