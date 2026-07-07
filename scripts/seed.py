"""Seed script: index the sample corpus (data/raw/ferry_docs) into Chroma.

The one command that makes a fresh checkout queryable:

    python scripts/seed.py            # local (needs OPENAI_API_KEY in .env)
    docker compose run --rm seed      # same, inside the compose stack

Idempotent: chunks upsert by stable content-hash IDs, so re-running never
duplicates. (The BM25 sparse index joins this step in V1.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    corpus = settings.corpus_dir
    if not corpus.is_dir():
        print(f"error: sample corpus not found at {corpus}", file=sys.stderr)
        return 1

    from rag.indexing import index_path  # local import: needs indexing extras

    print(f"Seeding from {corpus} ...")
    summary = index_path(corpus, settings=settings)
    print(
        f"Seeded {summary.chunks_indexed} chunks from {summary.files} file(s) "
        f"into '{settings.chroma_collection}' (total now {summary.total_chunks_in_store}; "
        f"bm25 {summary.bm25_chunks}; skipped {summary.chunks_skipped_duplicates} dupes)."
    )
    print(
        f"embedding_cost_usd={summary.embedding_cost_usd:.6f} timings_ms={summary.timings_ms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
