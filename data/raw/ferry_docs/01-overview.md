# Ferry — Overview

Ferry is an internal platform for ingesting files and running **asynchronous processing jobs** against them. You submit a job through the Ingest API or the `ferry` CLI, Ferry queues it, a worker executes it, and Ferry emits an event when the job completes or fails.

## What Ferry is for

Ferry is designed for batch and background file processing where you do not want to block on the result: cleaning datasets, transforming files, extracting data, and similar one-shot jobs. Work is durable — once a job is accepted it is queued and retried on failure rather than dropped.

## Core components

- **Ingest API** — REST endpoints to submit jobs (`POST /v1/jobs`) and track them (`GET /v1/jobs/{id}`).
- **Job Queue** — submitted jobs wait here until a worker is free.
- **Workers** — pull jobs from the queue and execute them. Each worker runs up to `ferry.worker.concurrency` jobs concurrently (default `4`).
- **Object Store** — holds both the input files for jobs and the outputs they produce.
- **Dead-Letter Queue (DLQ)** — jobs that fail after all retries are moved here so they can be inspected instead of lost.
- **Event Bus** — publishes `job.completed` and `job.failed` events that other systems can subscribe to.

## Typical flow

1. Submit a job (`POST /v1/jobs` or `ferry jobs submit`).
2. Ferry validates your API key and enqueues the job.
3. A worker pulls the job, reads its input from the object store, runs it, and writes the output back to the object store.
4. On success, Ferry emits `job.completed`. On failure, Ferry retries the job; if retries are exhausted it emits `job.failed` and moves the job to the DLQ.

## Where to go next

- Get running in five minutes → `02-quickstart.md`
- Every configuration key and its default → `03-configuration.md`
- What an error code means and how to fix it → `04-error-codes.md`
- Quotas and throttling → `05-rate-limits.md`
- Data flow, events, and terminology → `06-architecture.md`
