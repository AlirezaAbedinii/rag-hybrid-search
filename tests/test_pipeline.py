"""Phase 3 integration tests for the RAG pipeline (LLM mocked, no network)."""
from __future__ import annotations

import pytest

from rag.config import Settings
from rag.generation.llm_client import ChatResult
from rag.generation.prompts import REFUSAL_MESSAGE
from rag.indexing.vector_store import ScoredChunk
from rag.observability.metrics import Stopwatch, TokenUsage
from rag.pipeline import RAGPipeline


class FakeRetriever:
    """Returns preset chunks and records a 'dense' stage like the real one."""

    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results

    def retrieve(self, query, top_k=None, stopwatch: Stopwatch | None = None):
        if stopwatch is not None:
            with stopwatch.time("dense"):
                pass
        return list(self.results)


class FakeChat:
    """Scripted chat client that counts how many times it was called."""

    model = "fake-model"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        return ChatResult(text=self.text, usage=TokenUsage(prompt_tokens=100, completion_tokens=20))


def _settings(**kw) -> Settings:
    # Verification off by default here; the dedicated tests below turn it on.
    base = dict(retrieval_confidence_threshold=0.3, citation_verification=False)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _ctx(cid: str, text: str, score: float, source: str) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, text=text, score=score, metadata={"source_file": source})


def test_happy_path_generates_and_resolves_citations() -> None:
    contexts = [_ctx("c1", "Ferry retries failed jobs.", 0.9, "04-error-codes.md")]
    chat = FakeChat("Ferry retries failed jobs automatically [1].")
    pipe = RAGPipeline(FakeRetriever(contexts), chat, _settings(), mode="dense")

    res = pipe.answer("How does Ferry handle failures?")

    assert res.refused is False
    assert chat.calls == 1
    assert res.answer.endswith("[1].")
    assert len(res.citations) == 1
    assert res.citations[0].resolved and res.citations[0].chunk_id == "c1"
    assert res.retrieval_confidence == 0.9
    # Cost + latency instrumented on every request.
    assert res.usage.total_tokens == 120
    assert res.cost_usd > 0
    assert "generate" in res.timings_ms and "total_ms" in res.timings_ms


def test_empty_retrieval_triggers_i_dont_know_without_calling_llm() -> None:
    chat = FakeChat("should never be returned")
    pipe = RAGPipeline(FakeRetriever([]), chat, _settings(), mode="dense")

    res = pipe.answer("What is the meaning of life?")

    assert res.refused is True
    assert res.answer == REFUSAL_MESSAGE
    assert chat.calls == 0  # no fabrication, no generation cost
    assert res.citations == []
    assert res.retrieval_confidence == 0.0
    assert res.cost_usd == 0.0


def test_low_confidence_retrieval_refuses_without_calling_llm() -> None:
    weak = [_ctx("c1", "loosely related text", 0.1, "01-overview.md")]
    chat = FakeChat("should never be returned")
    pipe = RAGPipeline(FakeRetriever(weak), chat, _settings(), mode="dense")

    res = pipe.answer("Some off-topic question?")

    assert res.refused is True
    assert chat.calls == 0
    assert res.retrieval_confidence == pytest.approx(0.1)


def test_resolved_citations_only_reference_retrieved_chunks() -> None:
    contexts = [
        _ctx("c1", "alpha", 0.8, "a.md"),
        _ctx("c2", "beta", 0.7, "b.md"),
    ]
    chat = FakeChat("Combines [1] and [2], plus a bogus [9].")
    pipe = RAGPipeline(FakeRetriever(contexts), chat, _settings(), mode="dense")

    res = pipe.answer("Combine the sources.")

    valid_ids = {c.chunk_id for c in contexts}
    resolved = [c for c in res.citations if c.resolved]
    assert {c.chunk_id for c in resolved} == valid_ids
    assert any(not c.resolved for c in res.citations)  # [9] flagged, not invented


# --- citation verification + composite confidence (V1 behavior) ---------------
class ScriptedChat:
    """First complete() returns the answer; later calls return judge verdicts."""

    model = "fake-model"

    def __init__(self, answer: str, verdicts: list[str]) -> None:
        self.answer = answer
        self.verdicts = list(verdicts)
        self.calls = 0

    def complete(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(self.answer, TokenUsage(prompt_tokens=100, completion_tokens=20))
        return ChatResult(self.verdicts.pop(0), TokenUsage(prompt_tokens=30, completion_tokens=5))


def test_unsupported_citation_is_flagged_and_lowers_confidence() -> None:
    contexts = [
        _ctx("c1", "Ferry retries failed jobs three times.", 0.9, "04-error-codes.md"),
        _ctx("c2", "Workers pull jobs from the queue.", 0.85, "06-architecture.md"),
    ]
    answer = "Ferry retries failed jobs [1]. Ferry was founded in 1999 [2]."
    honest = ScriptedChat(answer, ["SUPPORTED", "SUPPORTED"])
    caught = ScriptedChat(answer, ["SUPPORTED", "UNSUPPORTED"])
    settings = _settings(citation_verification=True)

    good = RAGPipeline(FakeRetriever(contexts), honest, settings, mode="dense").answer("q")
    flagged = RAGPipeline(FakeRetriever(contexts), caught, settings, mode="dense").answer("q")

    # The fabricated claim's citation is flagged, not silently dropped...
    assert [c.supported for c in flagged.citations] == [True, False]
    # ...and the composite confidence drops relative to the fully-supported run.
    assert flagged.confidence < good.confidence
    assert flagged.confidence_breakdown["citation_coverage"] == 0.5
    # Judge calls: 1 generation + 2 verifications, usage/cost accounted.
    assert caught.calls == 3
    assert flagged.usage.total_tokens == 120 + 2 * 35
    assert "verify" in flagged.timings_ms


def test_confidence_lower_for_weak_retrieval_than_strong() -> None:
    answer = "Grounded claim [1]."
    strong_ctx = [_ctx("c1", "text", 0.95, "a.md")]
    weak_ctx = [_ctx("c1", "text", 0.45, "a.md")]
    settings = _settings(citation_verification=True)

    strong = RAGPipeline(
        FakeRetriever(strong_ctx), ScriptedChat(answer, ["SUPPORTED"]), settings, mode="dense"
    ).answer("q")
    weak = RAGPipeline(
        FakeRetriever(weak_ctx), ScriptedChat(answer, ["SUPPORTED"]), settings, mode="dense"
    ).answer("q")

    assert weak.confidence < strong.confidence
    assert weak.refused is False  # 0.45 clears the 0.3 gate; confidence reflects it


def test_verification_off_renormalizes_and_skips_judge() -> None:
    contexts = [_ctx("c1", "text", 0.9, "a.md")]
    chat = FakeChat("Claim [1].")
    pipe = RAGPipeline(FakeRetriever(contexts), chat, _settings(), mode="dense")

    res = pipe.answer("q")

    assert chat.calls == 1  # no judge calls
    assert res.confidence_breakdown["verified"] is False
    assert res.confidence_breakdown["citation_coverage"] is None
    assert all(c.supported is None for c in res.citations)
    assert res.confidence > 0  # renormalized, not dragged down by coverage
