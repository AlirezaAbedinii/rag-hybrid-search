"""Comparison reports: hybrid-vs-dense and chunking-strategy tables.

Runs the full golden-set evaluation under each configuration and renders the
markdown tables the README publishes (plan §6.3):

* **Retrieval modes** — ``dense`` vs ``hybrid`` over the same (fixed-strategy)
  index.
* **Chunking strategies** — ``fixed`` / ``recursive`` / ``semantic``, each
  ingested into its **own** Chroma collection and BM25 pickle
  (``<name>_<strategy>``) so runs never clobber each other.

Table building and rendering are pure functions over :class:`EvalReport`
aggregates, so tests exercise them with canned reports — the orchestration
(`main`) is the only part that needs API keys and a corpus.

Usage:
    python eval/compare.py                 # both experiments -> reports + tables
    python eval/compare.py --modes-only    # skip the (3x) chunking re-index runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.run_eval import EvalReport, evaluate, load_golden  # noqa: E402
from rag.config import Settings, get_settings  # noqa: E402

MODES = ("dense", "hybrid")
STRATEGIES = ("fixed", "recursive", "semantic")

# (label, aggregate key, format) rows shared by both tables.
_METRIC_ROWS = [
    ("Answer correctness (mean)", "correctness_mean", "{:.3f}"),
    ("Faithfulness (mean)", "faithfulness_mean", "{:.3f}"),
    ("Retrieval relevance", "retrieval_relevance_rate", "{:.3f}"),
    ("Citation accuracy", "citation_accuracy_mean", "{:.3f}"),
    ("Refusals (of no-answer set)", "refused", "{}"),
    ("Mean cost / query (USD)", "mean_cost_usd", "{:.6f}"),
    ("P95 total latency (ms)", "_p95_total", "{:.1f}"),
]


def _flat_aggregates(report: EvalReport) -> dict:
    """Aggregates plus derived keys the tables use."""
    agg = dict(report.aggregates)
    agg["_p95_total"] = agg.get("latency_ms", {}).get("total_ms", {}).get("p95", 0.0)
    return agg


def build_comparison_rows(reports: dict[str, EvalReport]) -> list[list[str]]:
    """Rows of [metric, value-per-column...] for the given {column: report}."""
    columns = list(reports)
    flat = {col: _flat_aggregates(reports[col]) for col in columns}
    rows = []
    for label, key, fmt in _METRIC_ROWS:
        rows.append([label] + [fmt.format(flat[col].get(key, 0)) for col in columns])
    return rows


def render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavored markdown table (deterministic)."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _settings_for_strategy(base: Settings, strategy: str) -> Settings:
    """Derive per-strategy settings: own collection + BM25 pickle, same corpus."""
    bm25 = Path(base.bm25_index_path)
    return base.model_copy(
        update={
            "chunk_strategy": strategy,
            "chroma_collection": f"{base.chroma_collection}_{strategy}",
            "bm25_index_path": bm25.with_name(f"{bm25.stem}_{strategy}{bm25.suffix}"),
        }
    )


def _evaluate_config(settings: Settings, mode: str, records) -> EvalReport:
    from rag.generation.llm_client import get_chat_client
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(settings, mode=mode)
    judge = get_chat_client(settings)
    return evaluate(records, pipeline, judge)


def run_mode_comparison(settings: Settings, records) -> dict[str, EvalReport]:
    """Evaluate dense vs hybrid over the same (already-seeded) index."""
    return {mode: _evaluate_config(settings, mode, records) for mode in MODES}


def run_chunking_comparison(
    settings: Settings, records, mode: str = "hybrid"
) -> dict[str, EvalReport]:
    """Re-index per strategy, then evaluate each with the given mode."""
    from rag.indexing import index_path

    reports: dict[str, EvalReport] = {}
    for strategy in STRATEGIES:
        strategy_settings = _settings_for_strategy(settings, strategy)
        summary = index_path(settings.corpus_dir, settings=strategy_settings)
        print(
            f"[chunking:{strategy}] indexed {summary.chunks_indexed} chunks "
            f"(bm25 {summary.bm25_chunks}, skipped {summary.chunks_skipped_duplicates})"
        )
        reports[strategy] = _evaluate_config(strategy_settings, mode, records)
    return reports


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Hybrid-vs-dense + chunking comparisons.")
    parser.add_argument("--golden", default=str(settings.golden_set_path))
    parser.add_argument("--modes-only", action="store_true", help="Skip chunking runs.")
    parser.add_argument(
        "--out-dir", default=str(_REPO_ROOT / "eval" / "reports"), help="Report directory."
    )
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []

    mode_reports = run_mode_comparison(settings, records)
    sections.append(
        "## Hybrid vs dense-only\n\n"
        + render_markdown_table(
            ["Metric", *(m.capitalize() for m in MODES)],
            build_comparison_rows(mode_reports),
        )
    )
    for mode, report in mode_reports.items():
        (out_dir / f"eval_{mode}.json").write_text(report.to_json(), encoding="utf-8")

    if not args.modes_only:
        chunk_reports = run_chunking_comparison(settings, records)
        sections.append(
            "## Chunking strategies\n\n"
            + render_markdown_table(
                ["Metric", *(s.capitalize() for s in STRATEGIES)],
                build_comparison_rows(chunk_reports),
            )
        )
        for strategy, report in chunk_reports.items():
            (out_dir / f"eval_chunking_{strategy}.json").write_text(
                report.to_json(), encoding="utf-8"
            )

    comparison_md = "# Comparison report\n\n" + "\n\n".join(sections) + "\n"
    out_path = out_dir / "comparison.md"
    out_path.write_text(comparison_md, encoding="utf-8")
    print("\n" + comparison_md)
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
