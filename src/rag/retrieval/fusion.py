"""Reciprocal Rank Fusion: merge ranked lists by rank position.

RRF combines rankings without comparing their raw scores (cosine similarity and
BM25 live on incomparable scales): each item earns ``weight / (k + rank)`` from
every list it appears in, where ``rank`` is 1-based position and ``k`` (default
60, from config) damps the head-of-list advantage.

Pure functions — deterministic, no network. Ties break by (score desc, id asc)
so output order is stable across runs.
"""
from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ranking of ``(id, rrf_score)``.

    ``rankings`` are best-first id lists (e.g. dense then sparse);
    ``weights`` defaults to 1.0 per list. Items missing from a list simply earn
    nothing from it.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    if k < 1:
        raise ValueError("k must be >= 1")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + position)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
