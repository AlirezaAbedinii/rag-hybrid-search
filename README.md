# RAG Hybrid Search

[![CI](https://github.com/AlirezaAbedinii/rag-hybrid-search/actions/workflows/ci.yml/badge.svg)](https://github.com/AlirezaAbedinii/rag-hybrid-search/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
![License](https://img.shields.io/badge/license-MIT-green)

> Hybrid-search RAG over technical docs, targeting **`TBD`% faithfulness** and
> **`TBD`% citation accuracy** on a hand-written golden eval suite, at
> **P95 `TBD` ms** and **$`TBD`/query**.
>
> **No number in this README is invented.** Every `TBD` is produced by the
> repo's own evaluation harness and instrumentation ([how to reproduce](#reproducing-the-numbers));
> they are filled in from the generated report as the measurement runs land.

A production-grade **Retrieval-Augmented Generation** service: it ingests
multi-format documentation, retrieves the most relevant passages, and generates
**grounded answers with inline `[n]` citations** — refusing to answer when the
retrieved context isn't strong enough rather than hallucinating. Every request
is instrumented for **per-stage latency and cost-per-query**.

## Architecture

```mermaid
flowchart LR
    A[Docs: md/txt/pdf/html] --> B[Loader + Normalizer]
    B --> C[Chunker]
    C --> D[Dedup]
    D --> E[(Chroma: dense)]
    D --> F[(BM25: sparse)]
    Q[Question] --> G[Dense retrieve]
    Q --> H[Sparse retrieve]
    E --> G
    F --> H
    G --> I[RRF fusion]
    H --> I
    I --> J[Rerank top-5]
    J --> K[Grounded generation + citations]
    K --> L[Citation verify + confidence]
    L --> M[Answer + citations + confidence]
    K -.-> N[(Trace: latency/tokens/cost)]
    classDef v1 stroke-dasharray: 6 4
    class D,F,H,I,J,L v1
```

**Solid nodes are implemented and tested; dashed nodes are the V1 track in
progress** (HTML loading, dedup, BM25 sparse retrieval, RRF fusion, cross-encoder
reranking, citation verification + composite confidence). The dense path — ingest
→ chunk → embed → Chroma → dense retrieve → confidence gate → grounded generation
with citations → trace — runs end to end today, served by FastAPI + Streamlit in
Docker.

## Why hybrid retrieval for technical docs

Dense (embedding) retrieval is excellent at *meaning*: "How do I install the
CLI?" finds the quickstart even with zero shared words. But technical docs are
full of **exact tokens** — error codes (`FERRY-429`), config keys
(`ferry.worker.concurrency`), function names — where embeddings blur exactly the
signal that matters, and classical keyword search (BM25) excels. Hybrid retrieval
runs both, merges the rankings with **Reciprocal Rank Fusion**, and lets a local
**cross-encoder** rerank the shortlist for precision.

That dense-vs-hybrid gap is this project's central, *measured* claim: the golden
set deliberately contains exact-token lookups where dense-only retrieval should
struggle, and the eval harness reports the difference (table below, `TBD` until
the V1 hybrid path lands).

## Quickstart (60 seconds)

Requires Docker + an OpenAI API key.

```bash
git clone https://github.com/AlirezaAbedinii/rag-hybrid-search.git && cd rag-hybrid-search
cp .env.example .env                 # put your OPENAI_API_KEY in .env
docker compose up -d --build         # API :8000, UI :8501
docker compose run --rm seed         # index the sample corpus (idempotent)

curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FERRY-429 mean?", "mode": "dense", "top_k": 5}'
```

Then open the UI at <http://localhost:8501> or the OpenAPI docs at
<http://localhost:8000/docs>. The `/v1/ask` response carries the answer, `[n]`
citations mapped to source chunks, the ranked retrieved contexts, a confidence
score, token usage, `cost_usd`, and per-stage `timings_ms`. Other endpoints:
`POST /v1/ingest`, `GET /v1/documents`, `GET /v1/stats`.

<details>
<summary>Local development without Docker</summary>

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e ".[dev]"
make test                                # deterministic suite; no key needed
python scripts/ingest.py --dry-run       # chunk the corpus; no key needed

pip install -e ".[ingestion,indexing,llm,api]"      # full local stack
cp .env.example .env                     # set OPENAI_API_KEY
python scripts/seed.py                   # build the index
make run-api                             # uvicorn on :8000
```
</details>

## Evaluation results

Quality is scored by a hand-built **LLM-as-judge** harness over a golden set of
hand-written Q/A pairs spanning four categories: direct lookups, multi-hop
questions, questions with **no answer in the corpus** (the system must refuse),
and ambiguous questions. Ground-truth answers are human-written and verified —
never LLM-generated. See [`eval/golden/SCHEMA.md`](eval/golden/SCHEMA.md).

> ⚠️ **Measurement status:** the harness (correctness + faithfulness) and the
> 15-question MVP golden set are implemented and CI-tested; the full scored run
> and the comparison experiments require the V1 features (hybrid retrieval,
> extra chunkers, retrieval-relevance + citation-accuracy metrics). Cells below
> are `TBD` until produced by `eval/run_eval.py` / `eval/compare.py` — they will
> be filled with the exact numbers the report emits.

### Hybrid vs dense-only

| Metric | Dense-only | Hybrid (RRF + rerank) |
|---|---|---|
| Answer correctness | `TBD` | `TBD` *(V1)* |
| Faithfulness | `TBD` | `TBD` *(V1)* |
| Retrieval relevance | `TBD` *(V1 metric)* | `TBD` *(V1)* |
| Citation accuracy | `TBD` *(V1 metric)* | `TBD` *(V1)* |
| Refusal correctness (no-answer set) | `TBD` | `TBD` *(V1)* |

### Chunking strategies

| Metric | Fixed (800/120) | Recursive *(V1)* | Semantic *(V1)* |
|---|---|---|---|
| Answer correctness | `TBD` | `TBD` | `TBD` |
| Faithfulness | `TBD` | `TBD` | `TBD` |
| Retrieval relevance | `TBD` | `TBD` | `TBD` |
| Cost per query | `TBD` | `TBD` | `TBD` |

### Reproducing the numbers

```bash
python eval/run_eval.py            # full suite -> eval/reports/latest.json + summary
python eval/run_eval.py --smoke    # mocked 3-case run (what CI executes; no API calls)
python eval/compare.py             # hybrid-vs-dense + chunking tables (V1)
```

## Latency & cost

Instrumentation is built in from the first request, not sampled after the fact:
every query records **per-stage latency** (`embed`, `dense`, `generate`, …
`total_ms`) and **token-based cost** (prompt/completion tokens × configured
prices) into a SQLite trace store. `GET /v1/stats` serves the rollup —
**P50/P95/P99 per stage** plus cost totals — and the Streamlit UI shows the same
per-request panel.

| Stage | P50 | P95 | P99 |
|---|---|---|---|
| embed | `TBD` | `TBD` | `TBD` |
| dense retrieve | `TBD` | `TBD` | `TBD` |
| generate | `TBD` | `TBD` | `TBD` |
| **total** | `TBD` | `TBD` | `TBD` |

**Cost per query:** mean `$TBD` / median `$TBD` (breakdown: embedding `$TBD`,
generation `$TBD`). Numbers are read from `GET /v1/stats` after a measurement
run over the golden set; they land here together with the eval tables.

## Design decisions

- **Chunking:** fixed-size sliding window (800 chars, 120 overlap) as the
  measured baseline; structure-aware recursive and semantic chunkers are V1,
  switchable via config, so the comparison is apples-to-apples.
- **Retrieval:** dense top-k over ChromaDB (cosine) today; V1 adds BM25 and
  **RRF** starting at 0.7 dense / 0.3 sparse (configurable), then a local
  cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranking top-20 → top-5 — precision
  without extra LLM spend.
- **"I don't know" policy:** if retrieval confidence falls below a threshold
  (default 0.30), the service returns a structured refusal **before** calling
  the LLM — no fabrication, no wasted generation cost. Refusing correctly is a
  *scored* behavior in the golden set, not an afterthought.
- **Citations are verifiable:** answers cite `[n]` against the exact numbered
  context they were generated from; citations that point outside the retrieved
  set are flagged, and V1 adds an LLM-as-judge pass confirming each cited chunk
  actually supports its claim.
- **One LLM provider** (OpenAI *or* Anthropic behind one interface) — no
  multi-provider routing. **File-based Chroma** — zero extra infrastructure.
  **Streamlit, not React; no auth; no streaming** — deliberately out of scope to
  keep the project focused and finishable.

## Project layout

```
src/rag/
├── config.py            # settings, model IDs, token prices, thresholds (env)
├── ingestion/           # loaders, normalizer, chunkers
├── indexing/            # embeddings client, Chroma wrapper, index_path()
├── retrieval/           # dense retrieval + dense/hybrid mode switch
├── generation/          # grounded prompt, LLM client, citation parsing
├── observability/       # per-stage timers, cost accounting, trace store
├── api/                 # FastAPI: /v1/ask /v1/ingest /v1/documents /v1/stats
└── pipeline.py          # retrieve → gate → generate → cite
eval/                    # golden set + LLM-as-judge harness + reports
ui/app.py                # Streamlit front end
scripts/                 # ingest.py (CLI) + seed.py (sample corpus)
tests/                   # deterministic tests (LLM mocked)
```

## Tests & CI

```bash
make test     # pytest — deterministic, no network, LLM mocked
make lint     # ruff
make eval     # full eval run (requires API key + seeded index)
```

CI (badge above) runs **ruff + pytest + a mocked eval smoke run** on every push —
no paid API calls in CI. The suite covers loaders/chunkers on real fixtures,
citation-parser edge cases, retrieval ranking with injected fakes, the refusal
gate, the API contract (happy path, validation, error surfaces), and the eval
harness with a scripted judge.

## Status & roadmap

Dense-only slice **done** end to end (ingest → retrieve → grounded generate →
API/UI → Docker → eval harness). In progress (V1): BM25 + RRF + reranker,
recursive/semantic chunkers + dedup, retrieval-relevance + citation-accuracy
metrics, citation verification + composite confidence, and the measured
comparison tables above. A demo storyboard lives in [`DEMO.md`](DEMO.md).

## License

[MIT](LICENSE)
