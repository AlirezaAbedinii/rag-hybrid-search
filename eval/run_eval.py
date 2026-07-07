"""Run the eval suite headlessly against eval/golden/golden_set.jsonl and write a report.

Pipeline per golden question:

    answerer.answer(question) -> AnswerResult
        -> correctness          (judge vs expected_answer; refusal check for no_answer)
        -> faithfulness         (judge answer vs retrieved context; skipped when refused)
        -> retrieval relevance  (deterministic: supporting_source in top-k)
        -> citation accuracy    (share of citations verified supported at answer time)

Results are aggregated per-metric and per-category, together with per-stage
latency percentiles (P50/P95/P99) and cost-per-query, and written to a JSON
report plus a printed summary.

Modes
-----
* ``--smoke`` — run ~3 cases through in-memory fakes (no network, no API key,
  no paid calls). This is what CI runs to prove the harness works end to end.
* default — run every golden question through the real ``RAGPipeline`` and a real
  judge (needs API keys + the index already seeded). Correctness ground truth is
  hand-written; only the grading is done by an LLM.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Support both `python eval/run_eval.py` and `import eval.run_eval`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.metrics import (  # noqa: E402
    Judge,
    MetricResult,
    refusal_correctness,
    score_citation_accuracy,
    score_correctness,
    score_faithfulness,
    score_retrieval_relevance,
)
from rag.config import Settings, get_settings  # noqa: E402
from rag.observability.metrics import percentiles  # noqa: E402


@dataclass(frozen=True)
class GoldenRecord:
    """One hand-written golden question (see eval/golden/SCHEMA.md)."""

    id: str
    question: str
    expected_answer: str
    supporting_sources: list[str]
    category: str
    notes: str = ""


@dataclass
class RecordEval:
    """The evaluation outcome for a single golden question."""

    id: str
    category: str
    refused: bool
    correctness: MetricResult
    faithfulness: MetricResult | None
    retrieval_relevance: MetricResult | None
    citation_accuracy: MetricResult | None
    cost_usd: float
    total_ms: float
    timings_ms: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """Full report: per-record results plus aggregate scores."""

    n: int
    aggregates: dict
    per_category: dict
    records: list[RecordEval] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "n": self.n,
                "aggregates": self.aggregates,
                "per_category": self.per_category,
                "records": [
                    {
                        "id": r.id,
                        "category": r.category,
                        "refused": r.refused,
                        "correctness": asdict(r.correctness),
                        "faithfulness": asdict(r.faithfulness) if r.faithfulness else None,
                        "retrieval_relevance": (
                            asdict(r.retrieval_relevance) if r.retrieval_relevance else None
                        ),
                        "citation_accuracy": (
                            asdict(r.citation_accuracy) if r.citation_accuracy else None
                        ),
                        "cost_usd": round(r.cost_usd, 6),
                        "total_ms": round(r.total_ms, 3),
                    }
                    for r in self.records
                ],
            },
            indent=2,
        )


def load_golden(path: str | Path) -> list[GoldenRecord]:
    """Load the JSONL golden set into typed records."""
    records: list[GoldenRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        records.append(
            GoldenRecord(
                id=raw["id"],
                question=raw["question"],
                expected_answer=raw["expected_answer"],
                supporting_sources=raw.get("supporting_sources", []),
                category=raw["category"],
                notes=raw.get("notes", ""),
            )
        )
    return records


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def evaluate(records: list[GoldenRecord], answerer, judge: Judge) -> EvalReport:
    """Score every record with the given answerer + judge and aggregate."""
    results: list[RecordEval] = []
    for rec in records:
        answer = answerer.answer(rec.question)
        context = "\n\n".join(c.text for c in answer.contexts)

        if rec.category == "no_answer":
            correctness = refusal_correctness(answer.refused)
        else:
            correctness = score_correctness(rec.question, rec.expected_answer, answer.answer, judge)

        # Faithfulness only applies when the system actually answered.
        faithfulness = (
            None if answer.refused else score_faithfulness(answer.answer, context, judge)
        )
        # Deterministic metrics: top-k hit + verified-citation share.
        relevance = score_retrieval_relevance(rec.supporting_sources, answer.contexts)
        accuracy = None if answer.refused else score_citation_accuracy(answer.citations)

        results.append(
            RecordEval(
                id=rec.id,
                category=rec.category,
                refused=answer.refused,
                correctness=correctness,
                faithfulness=faithfulness,
                retrieval_relevance=relevance,
                citation_accuracy=accuracy,
                cost_usd=answer.cost_usd,
                total_ms=answer.timings_ms.get("total_ms", 0.0),
                timings_ms=dict(answer.timings_ms),
            )
        )

    return _aggregate(results)


def _aggregate(results: list[RecordEval]) -> EvalReport:
    corr = [r.correctness.score for r in results]
    faith = [r.faithfulness.score for r in results if r.faithfulness is not None]
    relevance = [r.retrieval_relevance.score for r in results if r.retrieval_relevance]
    accuracy = [r.citation_accuracy.score for r in results if r.citation_accuracy]
    costs = [r.cost_usd for r in results]

    # Per-stage latency series across all evaluated questions -> P50/P95/P99.
    stage_series: dict[str, list[float]] = {}
    for r in results:
        for stage, ms in r.timings_ms.items():
            stage_series.setdefault(stage, []).append(float(ms))

    aggregates = {
        "correctness_mean": _mean(corr),
        "correctness_pass_rate": _mean([1.0 if r.correctness.passed else 0.0 for r in results]),
        "faithfulness_mean": _mean(faith),
        "faithfulness_pass_rate": _mean(
            [1.0 if r.faithfulness.passed else 0.0 for r in results if r.faithfulness]
        ),
        "retrieval_relevance_rate": _mean(relevance),
        "retrieval_relevance_n": len(relevance),
        "citation_accuracy_mean": _mean(accuracy),
        "citation_accuracy_n": len(accuracy),
        "answered": sum(1 for r in results if not r.refused),
        "refused": sum(1 for r in results if r.refused),
        "total_cost_usd": round(sum(costs), 6),
        "mean_cost_usd": round(statistics.fmean(costs), 6) if costs else 0.0,
        "median_cost_usd": round(statistics.median(costs), 6) if costs else 0.0,
        "mean_total_ms": _mean([r.total_ms for r in results]),
        "latency_ms": {
            stage: percentiles(series) for stage, series in sorted(stage_series.items())
        },
    }
    per_category: dict[str, dict] = {}
    for r in results:
        cat = per_category.setdefault(r.category, {"n": 0, "correctness_scores": []})
        cat["n"] += 1
        cat["correctness_scores"].append(r.correctness.score)
    for cat in per_category.values():
        cat["correctness_mean"] = _mean(cat.pop("correctness_scores"))
    return EvalReport(
        n=len(results), aggregates=aggregates, per_category=per_category, records=results
    )


def print_summary(report: EvalReport) -> None:
    agg = report.aggregates
    print(f"\nEvaluated {report.n} golden questions")
    print("-" * 48)
    print(
        f"  correctness   mean={agg['correctness_mean']:.3f}"
        f"  pass={agg['correctness_pass_rate']:.3f}"
    )
    print(
        f"  faithfulness  mean={agg['faithfulness_mean']:.3f}"
        f"  pass={agg['faithfulness_pass_rate']:.3f}"
    )
    print(
        f"  retrieval_relevance rate={agg['retrieval_relevance_rate']:.3f}"
        f" (n={agg['retrieval_relevance_n']})"
        f"  citation_accuracy mean={agg['citation_accuracy_mean']:.3f}"
        f" (n={agg['citation_accuracy_n']})"
    )
    print(f"  answered={agg['answered']}  refused={agg['refused']}")
    total_p = agg["latency_ms"].get("total_ms", {})
    print(
        f"  cost=${agg['total_cost_usd']:.6f} (mean ${agg['mean_cost_usd']:.6f}/q)"
        f"  latency p50={total_p.get('p50', 0):.1f}ms p95={total_p.get('p95', 0):.1f}ms"
    )
    print("  by category:")
    for cat, stats in sorted(report.per_category.items()):
        print(f"    {cat:<10} n={stats['n']:<3} correctness_mean={stats['correctness_mean']:.3f}")


# --- Smoke-mode fakes (no network) ----------------------------------------
def _run_smoke(records: list[GoldenRecord]) -> EvalReport:
    """Run a few records through in-memory fakes to prove the harness works."""
    from rag.generation.citations import Citation
    from rag.generation.llm_client import ChatResult
    from rag.indexing.vector_store import ScoredChunk
    from rag.observability.metrics import TokenUsage
    from rag.pipeline import AnswerResult

    subset = records[:3]

    class FakeAnswerer:
        def answer(self, question: str) -> AnswerResult:
            rec = next(r for r in subset if r.question == question)
            if rec.category == "no_answer":
                return AnswerResult(
                    question=question, answer="I don't know", mode="dense",
                    refused=True, retrieval_confidence=0.0,
                    timings_ms={"embed": 0.5, "dense": 0.5, "total_ms": 1.0},
                )
            source = rec.supporting_sources[0] if rec.supporting_sources else "doc.md"
            ctx = ScoredChunk("c1", "supporting context text", 0.9, {"source_file": source})
            return AnswerResult(
                question=question, answer=f"{rec.expected_answer} [1]", mode="dense",
                refused=False, retrieval_confidence=0.9, confidence=0.9,
                citations=[
                    Citation(
                        index=1, resolved=True, chunk_id="c1",
                        source_file=source, supported=True,
                    )
                ],
                contexts=[ctx], usage=TokenUsage(80, 20), cost_usd=0.00003,
                timings_ms={"embed": 1.0, "dense": 1.0, "generate": 3.0, "total_ms": 5.0},
            )

    class FakeJudge:
        def complete(self, system: str, user: str) -> ChatResult:
            return ChatResult("Rating: 5", TokenUsage(50, 5))

    return evaluate(subset, FakeAnswerer(), FakeJudge())


def _run_real(records: list[GoldenRecord], settings: Settings, mode: str) -> EvalReport:
    from rag.generation.llm_client import get_chat_client
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(settings, mode=mode)
    judge = get_chat_client(settings)  # single provider; judge shares the provider
    return evaluate(records, pipeline, judge)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite.")
    parser.add_argument("--smoke", action="store_true", help="Mock ~3 cases; no network.")
    parser.add_argument("--mode", default=settings.default_mode, help="Retrieval mode (real runs).")
    parser.add_argument(
        "--golden", default=str(settings.golden_set_path), help="Golden set JSONL path."
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "eval" / "reports" / "latest.json"),
        help="Where to write the JSON report.",
    )
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    report = _run_smoke(records) if args.smoke else _run_real(records, settings, args.mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json(), encoding="utf-8")
    print_summary(report)
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
