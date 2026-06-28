# Ferry — Error Codes

Every error Ferry returns includes a stable code of the form `FERRY-NNN`. Use the code to look up the cause and fix below.

## Authentication and access

- **FERRY-401 — Unauthorized.** Your API key is missing or invalid. *Fix:* run `ferry auth login` again with a valid key, or check the `Authorization` header on direct API calls.
- **FERRY-403 — Forbidden.** Your key is valid but lacks permission for this job type. *Fix:* request access to the job type in the console.

## Request errors

- **FERRY-404 — Not Found.** No job exists with the ID you requested. *Fix:* check the job ID returned at submission time.
- **FERRY-429 — Rate Limit Exceeded.** You have exceeded your submission quota. The response includes a `Retry-After` header (in seconds). The **server does not retry for you** — the CLI/client retries according to your `ferry.retry.*` settings. See `05-rate-limits.md`.
- **FERRY-503 — Service Unavailable.** The job queue is temporarily full. Ferry **retries these automatically**; no action is usually needed.

## Job execution failures

- **FERRY-1001 — Job Timeout.** The job ran longer than `ferry.job.timeout_seconds` and the worker killed it. *Fix:* increase `ferry.job.timeout_seconds` (see `03-configuration.md`) or make the job faster.
- **FERRY-1002 — Input Not Found.** The job's input file was not found in the object store. *Fix:* re-upload the input file and resubmit the job.
- **FERRY-1003 — Retries Exhausted.** The job failed `ferry.retry.max_attempts` times and was moved to the **dead-letter queue**. *Fix:* inspect the failed job in the DLQ to find the underlying error (see `06-architecture.md`).
