"""FastAPI app + routes.

``POST /v1/ask`` answers a question with citations, confidence, and latency/cost
metadata, and logs a full trace per request. OpenAPI docs are served at ``/docs``.

The app is built by :func:`create_app`, which accepts injectable factories so
tests can swap the pipeline/trace store for fakes. Real provider clients are
constructed **lazily on first use** — importing this module, serving ``/docs``,
and running ``/health`` all work with no API key configured; a missing key
surfaces as a clear 503 on the endpoints that need it.
"""
from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from ..config import ConfigError, Settings, get_settings
from ..observability.trace_store import TraceStore
from ..pipeline import AnswerResult, RAGPipeline
from .schemas import (
    AskRequest,
    AskResponse,
    CitationModel,
    ContextModel,
    DocumentInfo,
    DocumentsResponse,
    IngestRequest,
    IngestResponse,
    StatsResponse,
    UsageModel,
)

PipelineFactory = Callable[[str], RAGPipeline]
# indexer(path) -> IndexSummary-like; store_factory() -> object with list_sources/count.
Indexer = Callable[[str], object]
StoreFactory = Callable[[], object]


def _default_pipeline_factory(settings: Settings) -> PipelineFactory:
    def factory(mode: str) -> RAGPipeline:
        return RAGPipeline.from_settings(settings, mode=mode)

    return factory


def _to_response(result: AnswerResult) -> AskResponse:
    return AskResponse(
        question=result.question,
        answer=result.answer,
        mode=result.mode,
        refused=result.refused,
        confidence=result.retrieval_confidence,
        citations=[
            CitationModel(
                index=c.index,
                resolved=c.resolved,
                chunk_id=c.chunk_id,
                source_file=c.source_file,
                section_heading=c.section_heading,
            )
            for c in result.citations
        ],
        contexts=[
            ContextModel(chunk_id=c.chunk_id, text=c.text, score=c.score, metadata=c.metadata)
            for c in result.contexts
        ],
        usage=UsageModel(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        cost_usd=round(result.cost_usd, 6),
        timings_ms=result.timings_ms,
    )


def create_app(
    settings: Settings | None = None,
    *,
    pipeline_factory: PipelineFactory | None = None,
    trace_store: TraceStore | None = None,
    indexer: Indexer | None = None,
    store_factory: StoreFactory | None = None,
) -> FastAPI:
    """Build the FastAPI app with injectable dependencies (fakes in tests)."""
    settings = settings or get_settings()

    app = FastAPI(
        title="RAG Hybrid Search",
        version="0.1.0",
        description=(
            "Retrieval-Augmented Generation over technical docs: grounded answers "
            "with [n] citations, a confidence-gated refusal path, and per-request "
            "latency/cost metadata."
        ),
    )
    app.state.settings = settings
    app.state.pipeline_factory = pipeline_factory or _default_pipeline_factory(settings)
    app.state.trace_store = trace_store  # created lazily so imports touch no disk
    app.state.pipelines = {}  # mode -> RAGPipeline, built on first use
    app.state.indexer = indexer
    app.state.store_factory = store_factory

    def _get_indexer() -> Indexer:
        if app.state.indexer is None:
            from ..indexing import index_path

            app.state.indexer = lambda path: index_path(path, settings=settings)
        return app.state.indexer

    def _get_store():
        if app.state.store_factory is None:
            from ..indexing.vector_store import VectorStore

            app.state.store_factory = lambda: VectorStore.from_settings(settings)
        return app.state.store_factory()

    def _get_trace_store() -> TraceStore:
        if app.state.trace_store is None:
            app.state.trace_store = TraceStore(settings.trace_store_path)
        return app.state.trace_store

    def _get_pipeline(mode: str) -> RAGPipeline:
        if mode not in app.state.pipelines:
            try:
                app.state.pipelines[mode] = app.state.pipeline_factory(mode)
            except NotImplementedError as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc
            except ConfigError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                # e.g. hybrid mode before the BM25 index has been seeded.
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ImportError as exc:
                raise HTTPException(
                    status_code=503, detail=f"Server missing a dependency: {exc}"
                ) from exc
        return app.state.pipelines[mode]

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ask", response_model=AskResponse, tags=["query"])
    def ask(body: AskRequest) -> AskResponse:
        """Answer a question from the indexed docs, with citations + metadata."""
        mode = body.mode or settings.default_mode
        pipeline = _get_pipeline(mode)
        result = pipeline.answer(body.question, top_k=body.top_k)

        # Log the full trace (latency, tokens, cost) for /v1/stats and analysis.
        _get_trace_store().record(
            {
                "question": result.question,
                "mode": result.mode,
                "refused": result.refused,
                "confidence": result.retrieval_confidence,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "cost_usd": result.cost_usd,
                "timings_ms": result.timings_ms,
            }
        )
        return _to_response(result)

    @app.post("/v1/ingest", response_model=IngestResponse, tags=["index"])
    def ingest(body: IngestRequest) -> IngestResponse:
        """Index a file or directory (defaults to the sample corpus)."""
        from pathlib import Path

        target = Path(body.path) if body.path else settings.corpus_dir
        if not target.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {target}")
        try:
            summary = _get_indexer()(str(target))
        except ConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=503, detail=f"Server missing a dependency: {exc}"
            ) from exc
        return IngestResponse(
            files=summary.files,
            chunks_indexed=summary.chunks_indexed,
            total_chunks_in_store=summary.total_chunks_in_store,
            embedding_cost_usd=summary.embedding_cost_usd,
            bm25_chunks=getattr(summary, "bm25_chunks", 0),
            chunks_skipped_duplicates=getattr(summary, "chunks_skipped_duplicates", 0),
            timings_ms=summary.timings_ms,
        )

    @app.get("/v1/documents", response_model=DocumentsResponse, tags=["index"])
    def documents() -> DocumentsResponse:
        """List indexed source documents and their chunk counts."""
        try:
            store = _get_store()
        except ImportError as exc:
            raise HTTPException(
                status_code=503, detail=f"Server missing a dependency: {exc}"
            ) from exc
        sources = store.list_sources()
        return DocumentsResponse(
            documents=[DocumentInfo(source_file=s, chunks=n) for s, n in sources.items()],
            total_chunks=store.count(),
        )

    @app.get("/v1/stats", response_model=StatsResponse, tags=["ops"])
    def stats() -> StatsResponse:
        """Cost/latency summary (P50/P95/P99 per stage) over all logged requests."""
        return StatsResponse(**_get_trace_store().aggregates())

    return app


# uvicorn entrypoint: `uvicorn rag.api.main:app`
app = create_app()
