# RAG Hybrid Search

Production-grade Retrieval-Augmented Generation over internal/technical docs:
hybrid retrieval (dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank),
grounded answers with verified `[n]` citations and a confidence score, a
hand-built LLM-as-judge evaluation harness, and per-stage latency/cost
instrumentation.

> **Status: Phase 0 (scaffold).** Application logic lands in Phases 1–5. See
> [`PROJECT_IMPLEMENTATION_PLAN.md`](PROJECT_IMPLEMENTATION_PLAN.md) for the full
> plan and [`CLAUDE.md`](CLAUDE.md) for the working context. The final
> portfolio README (headline eval numbers, architecture diagram, quickstart) is
> written in Phase 6.

## Quickstart (dev)

```bash
cp .env.example .env          # fill in API key(s); choose the generation model
make install                  # pip install -e ".[dev]"   (requires Python 3.11+)
make test                     # run the test suite
make lint                     # ruff
```

## Tech stack

Python 3.11+ · ChromaDB · `rank_bm25` · Reciprocal Rank Fusion ·
cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) · OpenAI `text-embedding-3-small`
(offline swap available) · FastAPI · Streamlit · Docker.
