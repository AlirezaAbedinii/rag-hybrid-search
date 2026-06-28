# Ferry — Architecture & Glossary

## Data flow

```
submit ──▶ Ingest API ──▶ Job Queue ──▶ Worker ──▶ Object Store (output)
                                          │
                                   success │ failure
                                          ▼        ▼
                                  job.completed   retry … exhausted ──▶ DLQ + job.failed
```

When you submit a job, the Ingest API authenticates it and places it on the **Job Queue**. A **Worker** pulls the job, reads its input from the **Object Store**, executes it, and writes the output back to the Object Store. Ferry then publishes an event on the **Event Bus**.

## Components in detail

- **Ingest API** — the entry point. Validates the API key, enforces rate limits, and enqueues jobs.
- **Job Queue** — the buffer of accepted jobs waiting for a free worker.
- **Workers** — execute jobs. Each worker runs up to `ferry.worker.concurrency` jobs at once (default `4`). A worker that exceeds `ferry.job.timeout_seconds` on a job kills it and returns `FERRY-1001`.
- **Object Store** — durable storage for job **inputs and outputs**.
- **Dead-Letter Queue (DLQ)** — when a job exhausts `ferry.retry.max_attempts`, Ferry stops retrying, returns `FERRY-1003`, and moves the job here so the failure can be inspected without losing the job. Controlled by `ferry.dlq.enabled` (default `true`); set it to `false` to discard exhausted jobs instead of storing them.
- **Event Bus** — publishes job lifecycle events.

## Events

- **`job.completed`** — payload includes the job ID, output location, and duration.
- **`job.failed`** — payload includes the job ID, the last error code, and the attempt count.

## Glossary

- **Job** — a single unit of processing work submitted to Ferry.
- **Worker** — a process that pulls jobs from the queue and executes them.
- **Queue** — the buffer of submitted jobs waiting for a worker. This is **distinct from the dead-letter queue**, which holds permanently failed jobs.
- **Dead-Letter Queue (DLQ)** — storage for jobs that failed after all retries.
- **Object Store** — where job inputs and outputs are kept.
