# Ferry — Rate Limits

## Quota

Each API key may submit up to **100 jobs per minute**. Short bursts of up to **120 per minute** are tolerated, but sustained traffic above 100/min is throttled.

## What happens when you exceed the limit

Submissions over the limit are rejected with **`FERRY-429`**. The response includes a **`Retry-After`** header telling you how many seconds to wait before trying again.

## Retry behavior

Ferry does **not** automatically retry rate-limited submissions on the server side. Instead, the `ferry` CLI (and the official client) retries them for you using your retry configuration:

1. It waits for the duration in the `Retry-After` header.
2. It then retries up to `ferry.retry.max_attempts` times (default `3`).
3. It spaces those retries using the configured backoff — `ferry.retry.backoff` (default `exponential`, base `ferry.retry.base_delay_seconds` = `2` seconds).

See `03-configuration.md` for the retry keys.

## Raising your quota

Quota increases are requested through the Ferry console. There is **no configuration key** for the rate limit — it is enforced per API key on the server.
