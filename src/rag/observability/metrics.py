"""Per-stage timers and token + cost accounting.

Wired in from day one (not bolted on at the end). Two concerns live here:

* **Latency** — :class:`Stopwatch` records monotonic per-stage durations in ms.
  Stage names map to the query pipeline: ``embed``, ``dense``, ``sparse``,
  ``fusion``, ``rerank``, ``generate``, ``total``.
* **Cost** — :func:`token_cost` / :func:`generation_cost` turn token counts into
  USD using the prices configured in ``rag.config`` (per 1,000,000 tokens).

:func:`percentiles` (P50/P95/P99) is provided here so the eval report and the
``/v1/stats`` endpoint share one implementation. Everything is pure/deterministic
and has no network dependency.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

_PER_MILLION = 1_000_000


@dataclass
class Stopwatch:
    """Accumulates monotonic per-stage latencies in milliseconds."""

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Time a code block and add its duration (ms) to ``stage``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stages[stage] = self.stages.get(stage, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

    def as_dict(self) -> dict[str, float]:
        """Per-stage ms plus a ``total_ms`` rollup, rounded for reporting."""
        out = {k: round(v, 3) for k, v in self.stages.items()}
        out["total_ms"] = round(self.total_ms, 3)
        return out


@dataclass(frozen=True)
class TokenUsage:
    """Prompt/completion token counts for one provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def token_cost(n_tokens: int, usd_per_million: float) -> float:
    """USD cost of ``n_tokens`` at a per-1,000,000-token price."""
    return (n_tokens / _PER_MILLION) * usd_per_million


def generation_cost(
    usage: TokenUsage, input_price_per_million: float, output_price_per_million: float
) -> float:
    """USD cost of a generation call given its token usage and the two prices."""
    return token_cost(usage.prompt_tokens, input_price_per_million) + token_cost(
        usage.completion_tokens, output_price_per_million
    )


def percentiles(values: list[float], ps: tuple[int, ...] = (50, 95, 99)) -> dict[str, float]:
    """Nearest-rank percentiles, keyed ``p50``/``p95``/``p99``.

    Returns ``0.0`` for each percentile on an empty input so callers/reporting
    code don't have to special-case the no-data state.
    """
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for p in ps:
        # Nearest-rank: rank = ceil(p/100 * N), clamped to [1, N].
        rank = max(1, min(len(ordered), -(-p * len(ordered) // 100)))
        out[f"p{p}"] = round(ordered[rank - 1], 3)
    return out
