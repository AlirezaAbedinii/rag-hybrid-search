"""Write per-request traces (latency, tokens, cost) to SQLite.

Every ``/v1/ask`` request logs one trace row: question, mode, refusal flag,
confidence, token counts, cost, and the per-stage timings JSON. ``aggregates()``
rolls the rows up into the cost/latency summary served by ``GET /v1/stats``
(P50/P95/P99 per stage via ``rag.observability.metrics.percentiles``).

SQLite over JSONL because aggregation is a query, not a file scan — and it's
stdlib, so this adds no dependency. A connection is opened per operation, which
keeps the store trivially safe across threads/workers at this scale.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .metrics import percentiles

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    mode TEXT NOT NULL,
    refused INTEGER NOT NULL,
    confidence REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    timings_json TEXT NOT NULL
)
"""


class TraceStore:
    """Append-only SQLite store of per-request traces."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, trace: dict) -> None:
        """Insert one trace row (missing keys default sensibly)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO traces (ts, question, mode, refused, confidence,"
                " prompt_tokens, completion_tokens, cost_usd, timings_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.get("ts", time.time()),
                    trace.get("question", ""),
                    trace.get("mode", "dense"),
                    1 if trace.get("refused") else 0,
                    float(trace.get("confidence", 0.0)),
                    int(trace.get("prompt_tokens", 0)),
                    int(trace.get("completion_tokens", 0)),
                    float(trace.get("cost_usd", 0.0)),
                    json.dumps(trace.get("timings_ms", {})),
                ),
            )

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def aggregates(self) -> dict:
        """Cost/latency summary across all traces (shape of ``GET /v1/stats``)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT refused, prompt_tokens, completion_tokens, cost_usd,"
                " timings_json FROM traces"
            ).fetchall()

        n = len(rows)
        refused = sum(r[0] for r in rows)
        costs = [r[3] for r in rows]

        # Collect per-stage latency series across requests.
        stage_series: dict[str, list[float]] = {}
        for row in rows:
            for stage, ms in json.loads(row[4]).items():
                stage_series.setdefault(stage, []).append(float(ms))

        return {
            "requests": n,
            "refused": refused,
            "refusal_rate": round(refused / n, 4) if n else 0.0,
            "total_cost_usd": round(sum(costs), 6),
            "mean_cost_usd": round(sum(costs) / n, 6) if n else 0.0,
            "total_prompt_tokens": sum(r[1] for r in rows),
            "total_completion_tokens": sum(r[2] for r in rows),
            "latency_ms": {
                stage: {**percentiles(series), "n": len(series)}
                for stage, series in sorted(stage_series.items())
            },
        }
