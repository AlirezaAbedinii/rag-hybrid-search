# Demo Storyboard (< 4 minutes)

Scene-by-scene script for the demo video. Beats 1, 2, and 5 are recordable
against the current build; beats 3 and 4 exercise V1 features and are scripted
here so the recording plan is ready the moment they land — **record after V1**.

## Prep checklist (before recording)

- [ ] `.env` contains a working `OPENAI_API_KEY`; fresh `data/` (delete
      `data/chroma` for a clean first-ingest shot).
- [ ] `docker compose up -d --build` done **before** recording (skip build wait);
      stack healthy: `curl localhost:8000/health` → `{"status":"ok"}`.
- [ ] Screen layout: terminal on the left, browser on the right with two tabs —
      UI (<http://localhost:8501>) and API docs (<http://localhost:8000/docs>).
- [ ] Ask each scripted question once off-camera so responses are warm and the
      stats panel has data; then **reset the trace store** (delete
      `data/traces.sqlite`, restart API) so on-camera stats start clean.
- [ ] Questions come from the golden set (`eval/golden/golden_set.jsonl`) — the
      demo shows the *evaluated* behavior, not cherry-picked prompts.

---

## Scene 1 — Ingest the corpus (0:00 – 0:30)

**Terminal.**

```bash
docker compose run --rm seed
```

**Show:** the summary line — chunks indexed, files, embedding cost, per-stage
timings. Then:

```bash
curl -s localhost:8000/v1/documents | jq
```

**Say:** "One command indexes the sample corpus — seven markdown docs become ~37
chunks in a persistent Chroma index. Note we already know what ingestion *cost*:
every stage of this system is metered from day one."

## Scene 2 — Three kinds of questions (0:30 – 1:35)

**UI tab, dense mode, top-k 5.** Ask, in order:

1. **Lookup** (golden `q001`): *"What does the error code FERRY-429 mean?"*
   **Show:** grounded answer; click a `[n]` citation — it jumps to the exact
   source chunk with its similarity score.
2. **Multi-hop** (golden `q007`): *"If a job exceeds its execution time, what
   error is returned and how do I fix it?"*
   **Show:** answer synthesized from **two** documents (error codes +
   configuration), each claim carrying its own citation.
3. **No-answer** (golden `q013`): *"Does Ferry support deploying workers on
   Kubernetes?"*
   **Show:** the structured refusal card — and in the latency/cost panel, the
   generation stage is **absent**: below the confidence threshold the system
   refuses *before* calling the LLM.

**Say:** "Grounded answers with verifiable citations; multi-document synthesis;
and when the docs don't contain the answer, it says so — it doesn't invent one,
and it doesn't pay for a generation call to find out."

## Scene 3 — Citation verification catches a fabrication (1:35 – 2:15) — **[V1: record after citation verification lands]**

**Planned shot.** Ask a question where the model over-reaches (rehearse to find
one; alternatively lower `RETRIEVAL_CONFIDENCE_THRESHOLD` so a weak-context
answer slips through).

**Show:** the verification pass flagging an unsupported citation — the claim
whose cited chunk does **not** support it renders with a red *unsupported*
marker, and the composite confidence drops visibly.

**Say:** "Every claim–citation pair is re-checked by a judge model. A citation
that doesn't actually support its claim gets flagged instead of silently
shipping — and the answer's confidence reflects it."

*(Partial today: citations pointing outside the retrieved set are already
flagged red in the UI; the judge-based support check is the V1 piece.)*

## Scene 4 — Hybrid vs dense, the exact-token case (2:15 – 3:00) — **[V1: record after BM25 + RRF + reranker land]**

**Planned shot.** In the UI, ask golden `q002`: *"What is the default value of
ferry.worker.concurrency?"* in **dense** mode, then flip the toggle to
**hybrid** and re-ask.

**Show:** the retrieved-chunks panel reordering — BM25 pins the chunk containing
the literal key `ferry.worker.concurrency` to the top under hybrid; point at the
rank change and the answer's citation now hitting the exact configuration table.

**Say:** "Embeddings are great at meaning but blur exact tokens — error codes,
config keys. Hybrid runs BM25 alongside dense retrieval, fuses the rankings, and
reranks with a cross-encoder. On our eval set that's worth `TBD` points of
retrieval relevance over dense-only." *(Number comes from `eval/compare.py`; do
not invent it.)*

## Scene 5 — The latency/cost story (3:00 – 3:45)

**UI latency/cost panel + sidebar stats, then terminal:**

```bash
curl -s localhost:8000/v1/stats | jq
```

**Show:** per-request stage timings and dollar cost in the panel; then the
service rollup — **P50/P95/P99 per stage**, refusal rate, mean cost per query.

**Say:** "Every request logs a full trace — so 'how fast' and 'how much' are
queries, not guesses: P95 sits at `TBD` ms and a query costs about $`TBD`, with
the breakdown showing exactly which stage you'd optimize next." *(Read the real
numbers off the screen while recording.)*

## Close (3:45 – 3:55)

**Show:** README header — CI badge green, headline eval numbers.

**Say (§8.4 framing):** "A RAG system with hybrid search — dense plus BM25,
rank fusion, and reranking — hitting `TBD`% faithfulness and `TBD`% citation
accuracy on a hand-built golden eval suite, with per-stage latency and
cost-per-query instrumented end to end." *(Fill both numbers from the final
eval report before recording this line.)*

---

### Timing budget

| Scene | Content | Time |
|---|---|---|
| 1 | Seed + documents | 0:30 |
| 2 | Lookup / multi-hop / refusal | 1:05 |
| 3 | Citation verification catch *(V1)* | 0:40 |
| 4 | Hybrid vs dense toggle *(V1)* | 0:45 |
| 5 | Latency/cost panel + `/v1/stats` | 0:45 |
| — | Close on README numbers | 0:10 |
| | **Total** | **3:55** |
