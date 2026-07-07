"""CLI: ingest a folder or file into the index.

Runs the ingestion pipeline: load -> normalize -> chunk -> embed -> store in
Chroma, reporting chunk count, embedding cost, and per-stage latency.

Examples
--------
    # Dry run — load + chunk only (no network, no key needed); print counts.
    python scripts/ingest.py --dry-run

    # Full ingest of the sample corpus into Chroma.
    python scripts/ingest.py

    # Ingest a specific file or folder.
    python scripts/ingest.py path/to/docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/ingest.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.ingestion import build_chunks_for_dir, build_chunks_for_file  # noqa: E402
from rag.observability.metrics import Stopwatch  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest documents into the index.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(settings.corpus_dir),
        help="File or directory to ingest (default: the sample corpus).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + chunk only; skip embedding/storing (no network).",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write normalized documents to data/processed.",
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        parser.error(f"path does not exist: {target}")

    if args.dry_run:
        sw = Stopwatch()
        with sw.time("load_chunk"):
            if target.is_dir():
                chunks = build_chunks_for_dir(target, settings=settings, persist=args.persist)
            else:
                chunks = build_chunks_for_file(target, settings=settings, persist=args.persist)
        sources = sorted({c.source_file for c in chunks})
        print(
            f"Loaded + chunked {len(chunks)} chunks from {len(sources)} "
            f"file(s): {', '.join(sources)}"
        )
        print(f"[dry-run] no embedding/storing. timings_ms={sw.as_dict()}")
        return 0

    from rag.indexing import index_path  # local import: needs indexing extras

    summary = index_path(target, settings=settings, persist_processed=args.persist)
    print(
        f"Indexed {summary.chunks_indexed} chunks from {summary.files} file(s) "
        f"into '{settings.chroma_collection}' (total now {summary.total_chunks_in_store}; "
        f"bm25 {summary.bm25_chunks}; skipped {summary.chunks_skipped_duplicates} dupes)."
    )
    print(
        f"embedding_cost_usd={summary.embedding_cost_usd:.6f} timings_ms={summary.timings_ms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
