"""Tests for the composite confidence score (deterministic, no network).

The plan's specified behavior: the composite increases/decreases with each of
its inputs, stays in [0, 1], and renormalizes when a component is unavailable.
"""
from __future__ import annotations

import pytest

from rag.generation.confidence import (
    answer_completeness,
    composite_confidence,
    retrieval_confidence,
)


class _Scored:
    def __init__(self, score: float) -> None:
        self.score = score


# --- retrieval confidence (existing gate input) -------------------------------
def test_retrieval_confidence_top_score_clamped() -> None:
    assert retrieval_confidence([_Scored(0.4), _Scored(0.8)]) == 0.8
    assert retrieval_confidence([_Scored(1.7)]) == 1.0  # clamped
    assert retrieval_confidence([]) == 0.0


# --- completeness ---------------------------------------------------------------
def test_completeness_counts_cited_sentences() -> None:
    assert answer_completeness("Fact one [1]. Fact two [2].") == 1.0
    assert answer_completeness("Fact one [1]. Uncited aside.") == 0.5
    assert answer_completeness("No citations at all.") == 0.0
    assert answer_completeness("") == 0.0


# --- composite ---------------------------------------------------------------
def test_composite_monotonic_in_each_input() -> None:
    base = composite_confidence(0.5, 0.5, 0.5)
    assert composite_confidence(0.9, 0.5, 0.5) > base  # retrieval up
    assert composite_confidence(0.5, 0.9, 0.5) > base  # coverage up
    assert composite_confidence(0.5, 0.5, 0.9) > base  # completeness up
    assert composite_confidence(0.1, 0.5, 0.5) < base  # retrieval down
    assert composite_confidence(0.5, 0.1, 0.5) < base  # coverage down
    assert composite_confidence(0.5, 0.5, 0.1) < base  # completeness down


def test_composite_weighted_mean_exact() -> None:
    # (0.8*0.5 + 1.0*0.3 + 0.5*0.2) / 1.0 = 0.8
    assert composite_confidence(0.8, 1.0, 0.5, weights=(0.5, 0.3, 0.2)) == pytest.approx(0.8)


def test_composite_renormalizes_when_coverage_missing() -> None:
    # Verification off -> coverage None: weights renormalize over the rest,
    # so a strong answer isn't dragged to zero by an unmeasured component.
    score = composite_confidence(0.8, None, 0.8, weights=(0.5, 0.3, 0.2))
    assert score == pytest.approx(0.8)
    lower = composite_confidence(0.8, None, 0.1, weights=(0.5, 0.3, 0.2))
    assert lower < score


def test_composite_bounds() -> None:
    assert composite_confidence(0.0, 0.0, 0.0) == 0.0
    assert composite_confidence(1.0, 1.0, 1.0) == 1.0
    assert composite_confidence(2.0, 1.5, 1.5) == 1.0  # inputs clamped


def test_composite_all_missing_is_retrieval_only() -> None:
    assert composite_confidence(0.6, None, None) == pytest.approx(0.6)
