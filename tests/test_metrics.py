"""Phase 1 tests for the observability cost/latency helpers (deterministic)."""
from __future__ import annotations

from rag.observability.metrics import (
    Stopwatch,
    TokenUsage,
    generation_cost,
    percentiles,
    token_cost,
)


def test_token_cost_per_million() -> None:
    assert token_cost(1_000_000, 0.02) == 0.02
    assert token_cost(500_000, 0.60) == 0.30
    assert token_cost(0, 5.0) == 0.0


def test_generation_cost_splits_input_and_output_prices() -> None:
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    # 1M input @ 0.15 + 1M output @ 0.60 = 0.75
    assert generation_cost(usage, 0.15, 0.60) == 0.75
    assert usage.total_tokens == 2_000_000


def test_percentiles_nearest_rank() -> None:
    values = [float(x) for x in range(1, 101)]  # 1..100
    p = percentiles(values)
    assert p["p50"] == 50.0
    assert p["p95"] == 95.0
    assert p["p99"] == 99.0


def test_percentiles_empty_is_zeroed() -> None:
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_stopwatch_accumulates_stages() -> None:
    sw = Stopwatch()
    with sw.time("embed"):
        pass
    with sw.time("embed"):
        pass
    with sw.time("generate"):
        pass
    d = sw.as_dict()
    assert set(d) == {"embed", "generate", "total_ms"}
    assert d["total_ms"] >= 0.0
