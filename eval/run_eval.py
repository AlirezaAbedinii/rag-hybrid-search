"""Run the eval suite headlessly against eval/golden/golden_set.jsonl and write a report.

Pipeline per golden question:

    answerer.answer(question) -> AnswerResult
        -> correctness  (judge vs expected_answer; or refusal check for no_answer)
        -> faithfulness (judge answer vs retrieved context; skipped when refused)

Results are aggregated per-metric and per-category, together with the latency/cost
the pipeline already records, and written to a JSON report plus a printed summary.

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
    score_correctness,
    score_faithfulness,
)
from rag.config import Settings, get_settings  # noqa: E402


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
    cost_usd: float
    total_ms: float


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

        results.append(
            RecordEval(
                id=rec.id,
                category=rec.category,
                refused=answer.refused,
                correctness=correctness,
                faithfulness=faithfulness,
                cost_usd=answer.cost_usd,
                total_ms=answer.timings_ms.get("total_ms", 0.0),
            )
        )

    return _aggregate(results)


def _aggregate(results: list[RecordEval]) -> EvalReport:
    corr = [r.correctness.score for r in results]
    faith = [r.faithfulness.score for r in results if r.faithfulness is not None]
    aggregates = {
        "correctness_mean": _mean(corr),
        "correctness_pass_rate": _mean([1.0 if r.correctness.passed else 0.0 for r in results]),
        "faithfulness_mean": _mean(faith),
        "faithfulness_pass_rate": _mean(
            [1.0 if r.faithfulness.passed else 0.0 for r in results if r.faithfulness]
        ),
        "answered": sum(1 for r in results if not r.refused),
        "refused": sum(1 for r in results if r.refused),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "mean_total_ms": _mean([r.total_ms for r in results]),
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
    print(f"  answered={agg['answered']}  refused={agg['refused']}")
    print(f"  cost=${agg['total_cost_usd']:.6f}  mean_latency={agg['mean_total_ms']:.1f}ms")
    print("  by category:")
    for cat, stats in sorted(report.per_category.items()):
        print(f"    {cat:<10} n={stats['n']:<3} correctness_mean={stats['correctness_mean']:.3f}")


# --- Smoke-mode fakes (no network) ----------------------------------------
def _run_smoke(records: list[GoldenRecord]) -> EvalReport:
    """Run a few records through in-memory fakes to prove the harness works."""
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
                    refused=True, retrieval_confidence=0.0, timings_ms={"total_ms": 1.0},
                )
            ctx = ScoredChunk("c1", "supporting context text", 0.9, {"source_file": "doc.md"})
            return AnswerResult(
                question=question, answer=f"{rec.expected_answer} [1]", mode="dense",
                refused=False, retrieval_confidence=0.9, contexts=[ctx],
                usage=TokenUsage(80, 20), cost_usd=0.00003, timings_ms={"total_ms": 5.0},
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
