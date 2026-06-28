# Ferry — Quickstart

This guide takes you from nothing to a running job.

## 1. Install the CLI

```bash
pip install ferry-cli
```

Verify the install:

```bash
ferry --version
```

## 2. Authenticate

Create an API key in the Ferry console, then log in:

```bash
ferry auth login --api-key <YOUR_KEY>
```

This stores a token at `~/.ferry/credentials`. The CLI sends it as an `Authorization: Bearer <token>` header on every request. You only need to do this once per machine.

## 3. Submit your first job

```bash
ferry jobs submit --file ./input.csv --type csv-clean
```

The command returns a job ID, for example `job_a1b2c3`. The job is now queued.

## 4. Check job status

```bash
ferry jobs status job_a1b2c3
```

A job moves through these statuses: `queued` → `running` → `succeeded` (or `failed`).

## 5. Fetch job logs

```bash
ferry jobs logs job_a1b2c3
```

## Using the API directly

If you are not using the CLI, call the Ingest API yourself:

- `POST /v1/jobs` with an `Authorization: Bearer <token>` header and a JSON body `{ "file": "...", "type": "..." }`.
- `GET /v1/jobs/{id}` to poll status.

If a request comes back with an error code (for example `FERRY-401`), see `04-error-codes.md`.
