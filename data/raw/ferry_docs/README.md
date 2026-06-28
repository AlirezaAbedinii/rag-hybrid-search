# Sample Corpus — "Ferry" Internal Docs (synthetic)

This is a **synthetic documentation corpus** describing a **fictional internal service called Ferry**. It exists so the RAG demo and evaluation are **fully reproducible and self-contained** — no external data, no proprietary docs, no network dependencies beyond the LLM/embedding API.

## Why this corpus is shaped the way it is

It is deliberately engineered to exercise every part of the hybrid-search pipeline:

- **Exact technical tokens for BM25 / sparse retrieval** — config keys (`ferry.worker.concurrency`, `ferry.job.timeout_seconds`, `ferry.retry.max_attempts`) and error codes (`FERRY-429`, `FERRY-1001`, `FERRY-1003`). These are where **hybrid beats dense-only**: a question that names the exact key/code should retrieve the right chunk via BM25 even when semantic similarity alone is weak. This is the documented "hybrid > dense" example the implementation plan asks for.
- **Multi-hop content** — answers that require combining two documents (e.g., an error code's meaning in `04-error-codes.md` + the fix/config key in `03-configuration.md`).
- **A deliberate ambiguity** — Ferry has **two different timeouts** (`ferry.api.timeout_seconds` = 30s vs `ferry.job.timeout_seconds` = 900s). A bare question like "what is the default timeout?" is genuinely ambiguous, so the system should surface the ambiguity rather than guess.
- **Clear out-of-scope boundaries** — the docs intentionally never mention certain topics (Kubernetes deployment, Datadog metrics export, pricing) so "no-answer-in-corpus" questions are unambiguous.

## Files

| File | Purpose | Notable retrieval targets |
|---|---|---|
| `01-overview.md` | What Ferry is + components + flow | components, DLQ, object store |
| `02-quickstart.md` | Install, auth, submit, status | CLI commands, `pip install ferry-cli` |
| `03-configuration.md` | Full config-key reference | **all `ferry.*` keys + defaults**, the two timeouts |
| `04-error-codes.md` | Error code reference | **`FERRY-*` codes**, causes, fixes |
| `05-rate-limits.md` | Quotas + retry behavior | `FERRY-429`, `Retry-After`, client retry |
| `06-architecture.md` | Data flow, events, glossary | DLQ, events, queue-vs-DLQ distinction |

## How to use it

1. Place this folder at `data/raw/ferry-docs/` in the repo (matches the path in `PROJECT_IMPLEMENTATION_PLAN.md`).
2. `python scripts/seed.py` ingests it into Chroma + BM25.
3. `python eval/run_eval.py` runs the golden set in `eval/golden/golden_set.jsonl` against it.

## Note on multi-format loaders

The corpus is authored in Markdown (git-friendly and maintainable). To exercise the **PDF** and **HTML** loaders specifically, convert one or two files, e.g.:

```bash
pandoc 05-rate-limits.md -o 05-rate-limits.html      # HTML loader
pandoc 03-configuration.md -o 03-configuration.pdf   # PDF loader
```

If you convert a file, **either replace the `.md` or exclude it from ingest** so you don't index the same content twice (that would trigger the dedup path and split source attribution in the golden set).

> Everything about Ferry is invented for testing. It is not a real product.
