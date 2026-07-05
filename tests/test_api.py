"""Phase 5 tests for the FastAPI service (LLM mocked via a fake pipeline)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.config import Settings
from rag.generation.citations import Citation
from rag.generation.prompts import REFUSAL_MESSAGE
from rag.indexing.vector_store import ScoredChunk
from rag.observability.metrics import TokenUsage
from rag.observability.trace_store import TraceStore
from rag.pipeline import AnswerResult


def _answered(question: str) -> AnswerResult:
    ctx = ScoredChunk(
        "c1", "Ferry retries failed jobs.", 0.88, {"source_file": "04-error-codes.md"}
    )
    return AnswerResult(
        question=question,
        answer="Ferry retries failed jobs automatically [1].",
        mode="dense",
        refused=False,
        retrieval_confidence=0.88,
        citations=[
            Citation(index=1, resolved=True, chunk_id="c1", source_file="04-error-codes.md")
        ],
        contexts=[ctx],
        usage=TokenUsage(prompt_tokens=120, completion_tokens=30),
        cost_usd=0.000045,
        timings_ms={"embed": 12.0, "dense": 3.0, "generate": 400.0, "total_ms": 415.0},
    )


def _refused(question: str) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=REFUSAL_MESSAGE,
        mode="dense",
        refused=True,
        retrieval_confidence=0.05,
        timings_ms={"embed": 12.0, "dense": 3.0, "total_ms": 15.0},
    )


class FakePipeline:
    def __init__(self, mode: str, result_builder) -> None:
        self.mode = mode
        self._build = result_builder

    def answer(self, question: str, top_k=None) -> AnswerResult:
        result = self._build(question)
        result.mode = self.mode
        return result


@pytest.fixture
def client(tmp_path):
    """App wired with a fake pipeline + a real (temp) trace store."""
    settings = Settings(_env_file=None)

    def factory(mode: str):
        if mode == "hybrid":
            raise NotImplementedError("Hybrid retrieval is a V1 feature; use mode='dense'.")
        return FakePipeline(mode, _answered)

    app = create_app(
        settings,
        pipeline_factory=factory,
        trace_store=TraceStore(tmp_path / "traces.sqlite"),
    )
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_happy_path_returns_documented_schema(client: TestClient) -> None:
    resp = client.post("/v1/ask", json={"question": "How does Ferry handle failures?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"].endswith("[1].")
    assert body["refused"] is False
    assert body["mode"] == "dense"
    assert body["confidence"] == pytest.approx(0.88)
    # Citations resolve to retrieved chunks.
    assert body["citations"] == [
        {
            "index": 1,
            "resolved": True,
            "chunk_id": "c1",
            "source_file": "04-error-codes.md",
            "section_heading": None,
        }
    ]
    assert body["contexts"][0]["chunk_id"] == "c1"
    assert body["contexts"][0]["score"] == pytest.approx(0.88)
    # Latency/cost metadata fields (the acceptance criterion).
    assert body["cost_usd"] == pytest.approx(0.000045)
    assert body["timings_ms"]["total_ms"] == pytest.approx(415.0)
    assert body["usage"]["total_tokens"] == 150


def test_ask_refusal_shape(tmp_path) -> None:
    settings = Settings(_env_file=None)
    app = create_app(
        settings,
        pipeline_factory=lambda mode: FakePipeline(mode, _refused),
        trace_store=TraceStore(tmp_path / "t.sqlite"),
    )
    resp = TestClient(app).post("/v1/ask", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["citations"] == []
    assert body["usage"]["total_tokens"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing question
        {"question": ""},  # empty question
        {"question": "ok", "mode": "sparse"},  # invalid mode literal
        {"question": "ok", "top_k": 0},  # top_k below bound
        {"question": "ok", "top_k": 999},  # top_k above bound
    ],
)
def test_ask_bad_input_is_422(client: TestClient, payload: dict) -> None:
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_ask_hybrid_mode_is_501_until_v1(client: TestClient) -> None:
    resp = client.post("/v1/ask", json={"question": "anything", "mode": "hybrid"})
    assert resp.status_code == 501
    assert "V1" in resp.json()["detail"]


def test_ask_logs_a_trace_per_request(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    app = create_app(
        Settings(_env_file=None),
        pipeline_factory=lambda mode: FakePipeline(mode, _answered),
        trace_store=store,
    )
    client = TestClient(app)
    client.post("/v1/ask", json={"question": "q one"})
    client.post("/v1/ask", json={"question": "q two"})
    assert store.count() == 2


def test_openapi_docs_are_exposed(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/ask" in schema["paths"]
