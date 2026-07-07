"""Pydantic request/response models for the API.

These define the public contract of the service (and drive the OpenAPI docs at
``/docs``). ``AskResponse`` mirrors :class:`rag.pipeline.AnswerResult`: the
answer text, the parsed ``[n]`` citations, the ranked retrieved contexts, the
confidence signal, and the latency/cost metadata recorded for every request.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body of ``POST /v1/ask``."""

    question: str = Field(min_length=1, max_length=2000, description="The user question.")
    mode: Literal["dense", "hybrid"] | None = Field(
        default=None,
        description="Retrieval mode. Defaults to the server's configured mode "
        "(dense). 'hybrid' returns 501 until the V1 retrieval stack lands.",
    )
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Chunks to retrieve (default from config)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What does FERRY-429 mean?", "mode": "dense", "top_k": 5}]
        }
    }


class CitationModel(BaseModel):
    """One ``[n]`` citation parsed from the answer, mapped to its source chunk."""

    index: int = Field(description="The n in [n], 1-based, as written in the answer.")
    resolved: bool = Field(description="False if [n] points outside the retrieved set.")
    chunk_id: str = ""
    source_file: str = ""
    section_heading: str | None = None


class ContextModel(BaseModel):
    """One retrieved chunk, ranked by similarity."""

    chunk_id: str
    text: str
    score: float = Field(description="Similarity score (1 - cosine distance).")
    metadata: dict = Field(default_factory=dict)


class UsageModel(BaseModel):
    """Token usage for the generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    """Response of ``POST /v1/ask``."""

    question: str
    answer: str
    mode: str
    refused: bool = Field(description="True when the system declined to answer.")
    confidence: float = Field(
        description="Retrieval confidence in [0,1] (composite confidence is V1)."
    )
    citations: list[CitationModel] = Field(default_factory=list)
    contexts: list[ContextModel] = Field(
        default_factory=list, description="Retrieved chunks, ranked by score."
    )
    usage: UsageModel = Field(default_factory=UsageModel)
    cost_usd: float = Field(description="Embedding + generation cost of this request.")
    timings_ms: dict[str, float] = Field(
        default_factory=dict, description="Per-stage latency (embed, dense, generate, total_ms)."
    )


class IngestRequest(BaseModel):
    """Body of ``POST /v1/ingest``."""

    path: str | None = Field(
        default=None,
        description="File or directory to ingest. Defaults to the sample corpus.",
    )


class IngestResponse(BaseModel):
    """Result of an ingest run."""

    files: int
    chunks_indexed: int
    total_chunks_in_store: int
    embedding_cost_usd: float
    bm25_chunks: int = Field(
        default=0, description="Chunks in the BM25 sparse index (kept in sync with Chroma)."
    )
    chunks_skipped_duplicates: int = Field(
        default=0, description="Near-duplicate chunks skipped by dedup (cosine > threshold)."
    )
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """One indexed source document."""

    source_file: str
    chunks: int


class DocumentsResponse(BaseModel):
    """Response of ``GET /v1/documents``."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total_chunks: int = 0


class StatsResponse(BaseModel):
    """Response of ``GET /v1/stats`` — cost/latency summary over all traces."""

    requests: int
    refused: int
    refusal_rate: float
    total_cost_usd: float
    mean_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int
    latency_ms: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Per-stage P50/P95/P99 (+ sample count n), e.g. {'generate': {'p50': ...}}.",
    )
