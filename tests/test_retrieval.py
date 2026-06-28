"""Phase 2 tests for dense retrieval + the mode switch (deterministic, no network).

Uses in-memory fakes — a bag-of-words embedder and a cosine vector store — so the
retriever is exercised end-to-end without Chroma or any API call.
"""
from __future__ import annotations

import math

import pytest

from rag.config import Settings
from rag.indexing.vector_store import ScoredChunk
from rag.observability.metrics import Stopwatch
from rag.retrieval import DenseRetriever, build_retriever

VOCAB = ["ferry", "rate", "limit", "timeout", "worker", "retry"]


def _embed(text: str) -> list[float]:
    """Deterministic bag-of-words vector over a fixed vocabulary."""
    low = text.lower()
    return [float(low.count(word)) for word in VOCAB]


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return _embed(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_embed(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class FakeStore:
    """In-memory cosine store implementing the SupportsQuery protocol."""

    def __init__(self, docs: dict[str, str]) -> None:
        self._docs = docs
        self._vectors = {cid: _embed(text) for cid, text in docs.items()}

    def query(self, query_embedding: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(
                chunk_id=cid,
                text=self._docs[cid],
                score=_cosine(query_embedding, vec),
                metadata={"source_file": f"{cid}.md"},
            )
            for cid, vec in self._vectors.items()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


CORPUS = {
    "gold": "FERRY-429 means rate limit exceeded, with a Retry-After header.",
    "timeout": "FERRY-1001 job timeout: the worker killed a slow job.",
    "worker": "Each worker runs several jobs with bounded concurrency.",
}


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, default_mode="dense", top_k=10)


def test_dense_retriever_returns_gold_in_top_k(settings: Settings) -> None:
    retriever = build_retriever(
        "dense", settings=settings, embedder=FakeEmbedder(), store=FakeStore(CORPUS)
    )
    results = retriever.retrieve("what does the rate limit error mean", top_k=2)
    assert results[0].chunk_id == "gold"
    assert len(results) == 2
    # Scores are returned and ranked in descending order.
    assert results[0].score >= results[1].score
    assert results[0].score > 0


def test_top_k_is_respected(settings: Settings) -> None:
    retriever = build_retriever(
        "dense", settings=settings, embedder=FakeEmbedder(), store=FakeStore(CORPUS)
    )
    assert len(retriever.retrieve("worker timeout", top_k=1)) == 1
    assert len(retriever.retrieve("worker timeout", top_k=99)) == len(CORPUS)


def test_default_top_k_comes_from_settings() -> None:
    settings = Settings(_env_file=None, top_k=2)
    retriever = build_retriever(
        "dense", settings=settings, embedder=FakeEmbedder(), store=FakeStore(CORPUS)
    )
    assert retriever.default_top_k == 2
    assert len(retriever.retrieve("rate limit")) == 2


def test_stopwatch_records_embed_and_dense_stages(settings: Settings) -> None:
    retriever = build_retriever(
        "dense", settings=settings, embedder=FakeEmbedder(), store=FakeStore(CORPUS)
    )
    sw = Stopwatch()
    retriever.retrieve("rate limit", stopwatch=sw)
    assert "embed" in sw.stages
    assert "dense" in sw.stages


def test_mode_switch_routes_dense(settings: Settings) -> None:
    retriever = build_retriever(
        "dense", settings=settings, embedder=FakeEmbedder(), store=FakeStore(CORPUS)
    )
    assert isinstance(retriever, DenseRetriever)
    assert retriever.mode == "dense"


def test_hybrid_mode_is_not_yet_implemented(settings: Settings) -> None:
    with pytest.raises(NotImplementedError):
        build_retriever("hybrid", settings=settings)


def test_unknown_mode_raises(settings: Settings) -> None:
    with pytest.raises(ValueError):
        build_retriever("sparse", settings=settings)
