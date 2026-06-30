"""Pipeline orchestration: retrieve -> generate -> score, with full trace logging.

:class:`RAGPipeline` ties the MVP answer path together:

    retrieve -> retrieval-confidence gate -> grounded generation -> parse citations

If retrieval is empty or its confidence is below
``settings.retrieval_confidence_threshold``, the pipeline **refuses** (returns the
structured "I don't know" result) **without calling the LLM** — no fabrication and
no generation cost. Otherwise it generates a grounded answer, parses ``[n]``
citations back to the retrieved chunks, and records per-stage latency + token cost
on every request. (Citation verification + composite confidence are V1.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings, get_settings
from .generation.citations import Citation, build_citations
from .generation.confidence import retrieval_confidence
from .generation.llm_client import ChatClient
from .generation.prompts import REFUSAL_MESSAGE, build_grounded_prompt
from .indexing.vector_store import ScoredChunk
from .observability.metrics import Stopwatch, TokenUsage, generation_cost


class SupportsRetrieve(Protocol):
    """A retriever the pipeline can drive (DenseRetriever satisfies this)."""

    def retrieve(
        self, query: str, top_k: int | None = ..., stopwatch: Stopwatch | None = ...
    ) -> list[ScoredChunk]: ...


@dataclass
class AnswerResult:
    """The full result of one ``/v1/ask``-style query, including trace metadata."""

    question: str
    answer: str
    mode: str
    refused: bool
    retrieval_confidence: float
    citations: list[Citation] = field(default_factory=list)
    contexts: list[ScoredChunk] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGPipeline:
    """Orchestrates retrieval + grounded generation for a single mode."""

    retriever: SupportsRetrieve
    chat_client: ChatClient
    settings: Settings
    mode: str = "dense"

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, mode: str | None = None
    ) -> RAGPipeline:
        """Wire a pipeline from real provider clients (needs keys + extras)."""
        settings = settings or get_settings()
        mode = mode or settings.default_mode
        from .generation.llm_client import get_chat_client
        from .retrieval import build_retriever

        return cls(
            retriever=build_retriever(mode, settings=settings),
            chat_client=get_chat_client(settings),
            settings=settings,
            mode=mode,
        )

    def answer(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Answer ``question``, refusing when retrieval confidence is too low."""
        sw = Stopwatch()
        embedder = getattr(self.retriever, "embedder", None)
        embed_cost_before = getattr(embedder, "total_cost_usd", 0.0)

        contexts = self.retriever.retrieve(question, top_k=top_k, stopwatch=sw)
        confidence = retrieval_confidence(contexts)
        embed_cost = getattr(embedder, "total_cost_usd", 0.0) - embed_cost_before

        # --- "I don't know" gate: refuse before spending a generation call ---
        if not contexts or confidence < self.settings.retrieval_confidence_threshold:
            return AnswerResult(
                question=question,
                answer=REFUSAL_MESSAGE,
                mode=self.mode,
                refused=True,
                retrieval_confidence=confidence,
                contexts=contexts,
                cost_usd=embed_cost,
                timings_ms=sw.as_dict(),
            )

        # --- Grounded generation ---
        system, user = build_grounded_prompt(question, contexts)
        with sw.time("generate"):
            result = self.chat_client.complete(system, user)
        citations = build_citations(result.text, contexts)
        gen_cost = generation_cost(
            result.usage,
            self.settings.price_generation_input_per_1m,
            self.settings.price_generation_output_per_1m,
        )
        return AnswerResult(
            question=question,
            answer=result.text,
            mode=self.mode,
            refused=False,
            retrieval_confidence=confidence,
            citations=citations,
            contexts=contexts,
            usage=result.usage,
            cost_usd=embed_cost + gen_cost,
            timings_ms=sw.as_dict(),
        )
