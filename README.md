# RAG Hybrid Search

[![CI](https://github.com/AlirezaAbedinii/rag-hybrid-search/actions/workflows/ci.yml/badge.svg)](https://github.com/AlirezaAbedinii/rag-hybrid-search/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade **Retrieval-Augmented Generation** service over technical
documentation. It ingests multi-format docs, retrieves the most relevant
passages, and generates **grounded answers with inline `[n]` citations** —
refusing to answer when the retrieved context isn't strong enough, rather than
hallucinating. Every request is instrumented for **per-stage latency and
cost-per-query**.

The system is built as a **dense-only vertical slice first** (working end to
end), then extended with hybrid retrieval, an evaluation harness, and a UI. See
[Project status](#project-status).

---

## Highlights

- **Grounded answers, real citations.** Answers are generated strictly from
  numbered retrieved context; every `[n]` is parsed back to the exact source
  chunk, and out-of-range citations are flagged rather than silently dropped.
- **Honest refusal.** A retrieval-confidence gate returns a structured
  "I don't know" *before* calling the LLM when context is weak — no fabrication,
  no wasted generation cost.
- **Cost & latency from day one.** Per-stage timers (`embed`, `dense`,
  `generate`, …) and token-based cost accounting are wired through the pipeline,
  not bolted on at the end.
- **Pluggable providers.** OpenAI by default, with an offline
  `sentence-transformers` embedding swap behind the same interface for
  zero-cost local runs.
- **Tested & linted.** Deterministic unit/integration tests with the LLM mocked
  (no network), enforced by `ruff` + `pytest` in CI.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[Docs: md / txt / pdf] --> B[Loader + Normalizer]
        B --> C[Fixed-size chunker]
        C --> D[Embeddings]
        D --> E[(ChromaDB)]
    end

    subgraph Query
        Q[Question] --> R[Dense retrieval top-k]
        E --> R
        R --> G{Confidence gate}
        G -- low --> IDK["I don't know"]
        G -- ok --> P[Grounded generation]
        P --> CIT[Citation parsing]
        CIT --> ANS["Answer + [n] citations"]
    end

    R -.-> M[(Per-stage latency / token cost)]
    P -.-> M
```

Planned **hybrid** path (see roadmap): BM25 sparse retrieval → Reciprocal Rank
Fusion → cross-encoder rerank, slotting in between retrieval and generation.

## Quickstart

Requires **Python 3.11+**.

```bash
git clone https://github.com/AlirezaAbedinii/rag-hybrid-search.git
cd rag-hybrid-search
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make test      # run the suite (deterministic, no network, no API key)
make lint      # ruff
```

### Ingesting documents

The repo ships with a small sample corpus of synthetic technical docs under
`data/raw/ferry_docs/`. Chunk it without any API key or heavy dependencies:

```bash
python scripts/ingest.py --dry-run      # load + chunk only; prints chunk count
```

For a real index (embeddings + Chroma), configure a key and install the
ingestion/indexing extras:

```bash
cp .env.example .env                     # set OPENAI_API_KEY
pip install -e ".[ingestion,indexing,llm]"
python scripts/ingest.py                 # embed the sample corpus into ChromaDB
```

### Asking a question (programmatic)

```python
from rag.pipeline import RAGPipeline

pipe = RAGPipeline.from_settings(mode="dense")
result = pipe.answer("How does the system handle job failures?")

print(result.answer)                 # grounded text with [n] citations
print(result.citations)              # each [n] mapped to its source chunk
print(result.refused)                # True -> structured "I don't know"
print(result.cost_usd, result.timings_ms)
```

A FastAPI `POST /v1/ask` endpoint and a Streamlit UI are on the roadmap below.

## Configuration

All settings load from the environment (or a `.env` file) via
[`src/rag/config.py`](src/rag/config.py); see [`.env.example`](.env.example) for
the full list. Importing config never requires a secret — keys are validated
lazily, only when a provider is actually called. Notable knobs:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` | `openai` | Pick one generation provider; offline embeddings available |
| `GENERATION_MODEL` / `EMBEDDING_MODEL` | `gpt-4o-mini` / `text-embedding-3-small` | Model IDs |
| `TOP_K` | `10` | Chunks retrieved per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Fixed-size chunker |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.30` | Below this → "I don't know" |
| `PRICE_*_PER_1M` | — | Token prices used for cost accounting |

## Project layout

```
src/rag/
├── config.py            # settings, model IDs, prices, thresholds (from env)
├── ingestion/           # loaders, normalizer, fixed-size chunker
├── indexing/            # embeddings client, ChromaDB wrapper
├── retrieval/           # dense retrieval + dense/hybrid mode switch
├── generation/          # grounded prompt, LLM client, citation parsing
├── observability/       # per-stage latency + token/cost accounting
└── pipeline.py          # retrieve → gate → generate → cite
eval/                    # golden set + evaluation harness (in progress)
scripts/ingest.py        # CLI: ingest a file/folder
tests/                   # deterministic tests (LLM mocked)
```

## Tech stack

Python 3.11+ · ChromaDB · OpenAI `text-embedding-3-small` (offline
`sentence-transformers` swap) · OpenAI / Anthropic for generation · `pydantic`
settings · `ruff` · `pytest`. Planned: `rank_bm25`, cross-encoder reranker,
FastAPI, Streamlit, Docker.

## Project status

The dense-only pipeline is implemented and tested end to end; the differentiated
features are in progress.

**Done**
- [x] Multi-format ingestion (Markdown, text, PDF) → normalize → fixed-size
      overlapping chunks with stable IDs and source/section/page metadata
- [x] Embeddings (OpenAI, offline swap) → persistent ChromaDB with idempotent
      upsert
- [x] Dense retrieval (top-k cosine) with a `dense` / `hybrid` mode switch
- [x] Grounded generation with inline `[n]` citations mapped to sources
- [x] Retrieval-confidence-gated "I don't know" refusal
- [x] Per-stage latency + token/cost instrumentation
- [x] Deterministic test suite + CI (ruff + pytest)

**Roadmap**
- [ ] Hybrid retrieval: BM25 sparse index + Reciprocal Rank Fusion +
      cross-encoder reranker (top-20 → top-5)
- [ ] Recursive + semantic chunkers (switchable) and near-duplicate dedup
- [ ] LLM-as-judge evaluation harness (correctness, faithfulness, retrieval
      relevance, citation accuracy) with hybrid-vs-dense and chunking comparisons
- [ ] Citation verification + composite confidence score
- [ ] FastAPI `POST /v1/ask` + Streamlit UI + Docker Compose

## Development

```bash
make test     # pytest (works without install via pythonpath=src)
make lint     # ruff check .
make install  # pip install -e ".[dev]"
```

CI runs `ruff` + `pytest` on every push. Unit tests are deterministic and mock
the LLM, so no paid API calls are made in CI.

## License

[MIT](LICENSE)
