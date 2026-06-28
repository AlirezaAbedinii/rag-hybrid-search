# Ferry — Configuration Reference

Ferry reads configuration from a `ferry.yaml` file in the working directory, or from environment variables prefixed with `FERRY_`. **Environment variables take precedence** over the file.

## Configuration keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ferry.api.base_url` | string | `https://api.ferry.internal` | Ingest API endpoint the client talks to. |
| `ferry.api.timeout_seconds` | int | `30` | **API request timeout** — how long the client waits for a single Ingest API response. |
| `ferry.worker.concurrency` | int | `4` | Number of jobs a single worker runs at the same time. |
| `ferry.job.timeout_seconds` | int | `900` | **Job execution timeout** — how long a worker lets a job run before killing it and returning `FERRY-1001`. |
| `ferry.retry.max_attempts` | int | `3` | Maximum retry attempts for a failed job before it is sent to the dead-letter queue. |
| `ferry.retry.backoff` | enum | `exponential` | Backoff strategy between retries: `fixed` or `exponential`. |
| `ferry.retry.base_delay_seconds` | int | `2` | Base delay used by the backoff strategy. |
| `ferry.dlq.enabled` | bool | `true` | Whether jobs that exhaust their retries are moved to the dead-letter queue. |

## Example `ferry.yaml`

```yaml
ferry:
  api:
    base_url: https://api.ferry.internal
    timeout_seconds: 30
  worker:
    concurrency: 4
  job:
    timeout_seconds: 900
  retry:
    max_attempts: 3
    backoff: exponential
    base_delay_seconds: 2
  dlq:
    enabled: true
```

## Environment variable mapping

Replace dots with underscores and uppercase the whole key. For example:

- `ferry.worker.concurrency` → `FERRY_WORKER_CONCURRENCY`
- `ferry.job.timeout_seconds` → `FERRY_JOB_TIMEOUT_SECONDS`

## A note on the two timeouts

Ferry has **two independent timeouts**, and they are easy to confuse:

- `ferry.api.timeout_seconds` (default `30`) controls how long the **client waits for an API response**.
- `ferry.job.timeout_seconds` (default `900`) controls how long a **worker lets a job run** before killing it and returning `FERRY-1001`.

These are unrelated. Changing one has no effect on the other. If you mean "my jobs are being cut off," that is `ferry.job.timeout_seconds`; if you mean "my API calls hang," that is `ferry.api.timeout_seconds`.
