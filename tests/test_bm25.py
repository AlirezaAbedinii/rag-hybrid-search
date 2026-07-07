"""Tests for the BM25 sparse index (deterministic, no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.bm25_index import BM25Index, tokenize
from rag.ingestion.chunkers import Chunk

pytest.importorskip("rank_bm25")


def _chunk(cid: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        chunk_id=cid, text=text, source_file=source, strategy="fixed", ordinal=0
    )


CORPUS = [
    _chunk("c-limits", "FERRY-429 Rate Limit Exceeded. The response includes Retry-After."),
    _chunk("c-config", "Set ferry.worker.concurrency to control jobs per worker (default 4)."),
    _chunk("c-arch", "Workers pull jobs from the queue and execute them asynchronously."),
    _chunk("c-dlq", "Failed jobs move to the dead-letter queue after retries are exhausted."),
]


# --- tokenizer -----------------------------------------------------------------
def test_tokenize_emits_plain_words() -> None:
    assert "rate" in tokenize("Rate Limit Exceeded")
    assert "429" in tokenize("FERRY-429")


def test_tokenize_keeps_compound_tokens_whole() -> None:
    tokens = tokenize("Set ferry.worker.concurrency and check FERRY-429.")
    assert "ferry.worker.concurrency" in tokens  # dotted config key intact
    assert "ferry-429" in tokens  # hyphenated error code intact


# --- exact-token retrieval (the reason BM25 exists here) -----------------------
def test_exact_error_code_ranks_its_chunk_first() -> None:
    index = BM25Index()
    index.upsert(CORPUS)
    results = index.query("What does FERRY-429 mean?", top_k=2)
    assert results[0][0] == "c-limits"


def test_exact_config_key_ranks_its_chunk_first() -> None:
    index = BM25Index()
    index.upsert(CORPUS)
    results = index.query("default value of ferry.worker.concurrency", top_k=2)
    assert results[0][0] == "c-config"


def test_query_shape_and_determinism() -> None:
    index = BM25Index()
    index.upsert(CORPUS)
    first = index.query("queue jobs", top_k=3)
    second = index.query("queue jobs", top_k=3)
    assert first == second
    cid, score, text, metadata = first[0]
    assert isinstance(score, float)
    assert text and metadata["source_file"] == "doc.md"


def test_empty_index_returns_empty() -> None:
    assert BM25Index().query("anything") == []


# --- upsert + sync semantics ----------------------------------------------------
def test_upsert_by_chunk_id_is_idempotent() -> None:
    index = BM25Index()
    index.upsert(CORPUS)
    index.upsert(CORPUS)  # re-ingest: same ids, no duplicates
    assert index.count() == len(CORPUS)


# --- persistence -----------------------------------------------------------------
def test_save_load_roundtrip(tmp_path: Path) -> None:
    index = BM25Index()
    index.upsert(CORPUS)
    index.save(tmp_path / "bm25.pkl")

    loaded = BM25Index.load(tmp_path / "bm25.pkl")
    assert loaded.count() == len(CORPUS)
    assert loaded.query("FERRY-429", top_k=1)[0][0] == "c-limits"


def test_load_missing_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed"):
        BM25Index.load(tmp_path / "missing.pkl")
    assert BM25Index.load_or_new(tmp_path / "missing.pkl").count() == 0
