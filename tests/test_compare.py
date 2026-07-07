"""Tests for the comparison report builders (deterministic, no network)."""
from __future__ import annotations

from eval.compare import (
    _settings_for_strategy,
    build_comparison_rows,
    render_markdown_table,
)
from eval.run_eval import EvalReport

from rag.config import Settings


def _report(correctness: float, relevance: float, cost: float, p95: float) -> EvalReport:
    return EvalReport(
        n=15,
        aggregates={
            "correctness_mean": correctness,
            "faithfulness_mean": 0.9,
            "retrieval_relevance_rate": relevance,
            "citation_accuracy_mean": 0.8,
            "refused": 3,
            "mean_cost_usd": cost,
            "latency_ms": {"total_ms": {"p50": p95 / 2, "p95": p95, "p99": p95 * 1.2}},
        },
        per_category={},
    )


def test_comparison_rows_pull_the_right_numbers_per_column() -> None:
    reports = {
        "dense": _report(0.70, 0.60, 0.0011, 900.0),
        "hybrid": _report(0.85, 0.90, 0.0013, 1400.0),
    }
    rows = build_comparison_rows(reports)
    by_label = {row[0]: row[1:] for row in rows}

    assert by_label["Answer correctness (mean)"] == ["0.700", "0.850"]
    assert by_label["Retrieval relevance"] == ["0.600", "0.900"]
    assert by_label["Mean cost / query (USD)"] == ["0.001100", "0.001300"]
    assert by_label["P95 total latency (ms)"] == ["900.0", "1400.0"]


def test_markdown_table_renders_github_flavored() -> None:
    md = render_markdown_table(["Metric", "Dense"], [["Faithfulness", "0.900"]])
    lines = md.splitlines()
    assert lines[0] == "| Metric | Dense |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| Faithfulness | 0.900 |"
    # Deterministic output.
    assert md == render_markdown_table(["Metric", "Dense"], [["Faithfulness", "0.900"]])


def test_settings_for_strategy_isolates_indexes() -> None:
    base = Settings(_env_file=None)
    derived = _settings_for_strategy(base, "recursive")

    assert derived.chunk_strategy == "recursive"
    assert derived.chroma_collection == f"{base.chroma_collection}_recursive"
    assert derived.bm25_index_path.name == "bm25_index_recursive.pkl"
    # The base settings are untouched (copy, not mutation).
    assert base.chunk_strategy == "fixed"
    assert base.chroma_collection == "ferry_docs"
