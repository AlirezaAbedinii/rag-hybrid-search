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


def test_unknown_mode_raises(settings: Settings) -> None:
    with pytest.raises(ValueError):
        build_retriever("sparse", settings=settings)


# --- hybrid: fakes for sparse + scorer ---------------------------------------
class FakeBM25:
    """Keyword index: exact token match on the fixed corpus."""

    def __init__(self, docs: dict[str, str]) -> None:
        self._docs = docs

    def query(self, text: str, top_k: int):
        tokens = set(text.lower().split())
        hits = [
            (cid, float(len(tokens & set(doc.lower().split()))), doc, {"source_file": f"{cid}.md"})
            for cid, doc in self._docs.items()
        ]
        hits = [h for h in hits if h[1] > 0]
        hits.sort(key=lambda h: (-h[1], h[0]))
        return hits[:top_k]


class KeywordScorer:
    """Deterministic 'cross-encoder': counts shared words with the query."""

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


HYBRID_CORPUS = {
    **CORPUS,
    # An exact-token chunk the bag-of-words dense vocab cannot see: 'XQJ-429'
    # shares no substring with VOCAB, so dense scores it 0 — but BM25 nails it.
    "exact": "XQJ-429 documentation lives here with the exact code token.",
}


def _hybrid(settings: Settings):
    return build_retriever(
        "hybrid",
        settings=settings,
        embedder=FakeEmbedder(),
        store=FakeStore(HYBRID_CORPUS),
        sparse_index=FakeBM25(HYBRID_CORPUS),
        scorer=KeywordScorer(),
    )


def test_hybrid_mode_builds_and_returns_top_k() -> None:
    settings = Settings(_env_file=None, rerank_top_k=2, rerank_candidates=10)
    results = _hybrid(settings).retrieve("worker timeout jobs")
    assert len(results) <= 2
    assert all(0.0 <= r.score <= 1.0 for r in results)  # sigmoid-normalized


def test_hybrid_surfaces_exact_token_that_dense_misses(settings: Settings) -> None:
    query = "worker XQJ-429 exact code"
    # Dense alone: the embedder has no signal for XQJ-429, so the 'exact'
    # chunk scores zero and is invisible among the semantic matches...
    dense_only = build_retriever(
        "dense", settings=settings, embedder=FakeEmbedder(), store=FakeStore(HYBRID_CORPUS)
    ).retrieve(query, top_k=3)
    dense_visible = {c.chunk_id for c in dense_only if c.score > 0}
    assert "worker" in dense_visible  # dense does see the semantic part
    assert "exact" not in dense_visible  # ...but not the exact code

    # ...while hybrid finds it via BM25 and the reranker promotes it to #1
    # (the documented BM25-beats-dense example).
    hybrid_ids = [c.chunk_id for c in _hybrid(settings).retrieve(query, top_k=3)]
    assert hybrid_ids[0] == "exact"


def test_hybrid_times_all_stages(settings: Settings) -> None:
    sw = Stopwatch()
    _hybrid(settings).retrieve("rate limit", stopwatch=sw)
    assert {"embed", "dense", "sparse", "fusion", "rerank"} <= set(sw.stages)


def test_rrf_weights_come_from_settings() -> None:
    settings = Settings(_env_file=None, rrf_dense_weight=0.9, rrf_sparse_weight=0.1, rrf_k=42)
    retriever = _hybrid(settings)
    assert retriever.dense_weight == 0.9
    assert retriever.sparse_weight == 0.1
    assert retriever.rrf_k == 42


def test_reranker_improves_gold_position() -> None:
    from rag.retrieval.rerank import Reranker

    gold = ScoredChunk("gold", "retry after rate limit exceeded", 0.2, {})
    noise = ScoredChunk("noise", "unrelated worker text", 0.9, {})
    reranked = Reranker(scorer=KeywordScorer(), top_k=2).rerank(
        "rate limit retry", [noise, gold]  # gold arrives ranked LAST
    )
    assert reranked[0].chunk_id == "gold"  # cross-encoder promotes it to #1


def test_reranker_empty_candidates() -> None:
    from rag.retrieval.rerank import Reranker

    assert Reranker(scorer=KeywordScorer(), top_k=5).rerank("q", []) == []
