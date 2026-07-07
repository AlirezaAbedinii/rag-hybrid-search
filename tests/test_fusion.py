"""Tests for Reciprocal Rank Fusion — exact math on hand-constructed lists.

Deterministic, no LLM, no network (the plan's Phase 2 test requirement).
"""
from __future__ import annotations

import pytest

from rag.retrieval.fusion import rrf_fuse


def test_rrf_math_is_exact() -> None:
    # Hand-computed with k=60, equal weights:
    #   a: dense#1 + sparse#2 -> 1/61 + 1/62
    #   b: dense#2 + sparse#1 -> 1/62 + 1/61   (tie with a)
    #   c: dense#3 only       -> 1/63
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)


def test_rrf_appearing_in_both_lists_beats_single_list() -> None:
    ranking = [i for i, _ in rrf_fuse([["a", "b"], ["b", "z"]], k=60)]
    assert ranking[0] == "b"  # #2 dense + #1 sparse > #1 dense alone


def test_rrf_weights_shift_the_ranking_predictably() -> None:
    dense = ["d-first", "both"]
    sparse = ["s-first", "both"]

    dense_heavy = [i for i, _ in rrf_fuse([dense, sparse], weights=[0.9, 0.1], k=60)]
    sparse_heavy = [i for i, _ in rrf_fuse([dense, sparse], weights=[0.1, 0.9], k=60)]

    # 'both' is #2 everywhere and always wins; the #1s swap with the weights.
    assert dense_heavy.index("d-first") < dense_heavy.index("s-first")
    assert sparse_heavy.index("s-first") < sparse_heavy.index("d-first")


def test_rrf_is_deterministic_with_stable_tie_break() -> None:
    lists = [["x", "y"], ["y", "x"]]  # x and y tie exactly
    first = rrf_fuse(lists, k=60)
    assert first == rrf_fuse(lists, k=60)
    assert [i for i, _ in first] == ["x", "y"]  # tie broken by id, stable


def test_rrf_k_damps_head_advantage() -> None:
    # With tiny k, rank #1 dominates; with huge k, scores flatten.
    small_k = dict(rrf_fuse([["top", "second"]], k=1))
    large_k = dict(rrf_fuse([["top", "second"]], k=10_000))
    assert small_k["top"] / small_k["second"] > large_k["top"] / large_k["second"]


def test_rrf_input_validation() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)


def test_rrf_empty_inputs() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
