"""Per-stage timers and token + cost accounting.

Stages: embed, dense, sparse, fusion, rerank, generate, total. Cost = tokens x configured price.
Lightweight, wired in from day one (not bolted on at the end)."""
