# Golden Evaluation Set — Schema & Guide

The golden set (`golden_set.jsonl`) is the hand-verified ground truth the evaluation harness scores against. **One JSON object per line** (JSONL). It pairs with the synthetic corpus in `data/raw/ferry-docs/`.

## Schema

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Stable unique ID (`q001`, `q002`, …). Never reuse an ID. |
| `question` | string | yes | The question as a user would ask it. |
| `expected_answer` | string | yes | The **hand-written** ideal answer. Used by the LLM-as-judge correctness metric. |
| `supporting_sources` | string[] | yes | Source files (optionally `file#section`) that contain the answer. **Empty `[]` for `no_answer`.** Used by the retrieval-relevance metric. |
| `category` | string | yes | One of `lookup`, `multi_hop`, `no_answer`, `ambiguous`. |
| `notes` | string | yes | Why this case exists / what it tests. Helps you keep coverage balanced as you grow the set. |

### Minimal example

```json
{"id": "q001", "question": "What does FERRY-429 mean?", "expected_answer": "Rate limit exceeded; the response includes a Retry-After header.", "supporting_sources": ["04-error-codes.md", "05-rate-limits.md"], "category": "lookup", "notes": "Exact error code → BM25 should win."}
```

## Categories and how each is scored

| Category | Maps to (plan §6.1) | Ideal behavior | How it's judged |
|---|---|---|---|
| `lookup` | Straightforward lookups | Answer directly from one chunk, with a citation. | Correctness vs `expected_answer`; retrieval relevance = was a `supporting_source` in top-k. |
| `multi_hop` | Multi-hop | Combine 2+ documents into one grounded answer. | Same metrics; tests whether fusion/rerank surfaced **all** needed sources. |
| `no_answer` | No-answer-in-corpus | **Refuse**: say it's not in the docs. Do **not** hallucinate. | Correctness = did it correctly decline. `supporting_sources` is `[]`. |
| `ambiguous` | Ambiguous | **Surface the ambiguity / ask to clarify** rather than guess one interpretation. | Correctness = did it name the competing interpretations instead of committing to one. |

## Rules (do not break these)

1. **Ground truth is hand-written.** Never generate `expected_answer` with an LLM — the whole point is human-verified truth. (You may use an LLM to *draft question phrasings*, but you verify every answer yourself against the corpus.)
2. **Keep `supporting_sources` accurate.** The retrieval-relevance metric depends on it. If you edit the corpus, re-check every affected row.
3. **`no_answer` rows must stay genuinely absent** from the corpus. If you later add a doc that answers one, change its category or remove it.
4. **One ID forever.** Add new IDs; don't renumber.

## Growing from the MVP set (15) to the V1 set (50+)

This starter set is the MVP (15 questions: 6 `lookup`, 4 `multi_hop`, 2 `ambiguous`, 3 `no_answer`). To reach the 50+ V1 set, keep a rough balance and prioritize:

- **Failure cases you discover during testing.** When the system gets something wrong, add that exact question — this is the highest-value way to grow the set.
- **More exact-token lookups** (other `ferry.*` keys, other `FERRY-*` codes) to strengthen the hybrid-vs-dense comparison.
- **Deeper multi-hop chains** (3 sources) to stress fusion + reranking.
- **More refusal cases** in categories where the LLM is tempted to hallucinate (pricing, security/compliance, integrations).
- **Near-miss ambiguities** — questions that look ambiguous but have a single correct reading, to check the system doesn't over-clarify.

## Suggested record template (copy when adding)

```json
{"id": "qNNN", "question": "", "expected_answer": "", "supporting_sources": [], "category": "lookup", "notes": ""}
```
