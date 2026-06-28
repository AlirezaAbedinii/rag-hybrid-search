"""CLI: ingest a folder or file into the index.

Runs the MVP ingestion pipeline: load -> normalize -> chunk -> embed -> store in
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


def _build_chunks(target: Path, persist: bool):
    settings = get_settings()
    if target.is_dir():
        return build_chunks_for_dir(target, settings=settings, persist=persist)
    return build_chunks_for_file(target, settings=settings, persist=persist)


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

    sw = Stopwatch()
    with sw.time("load_chunk"):
        chunks = _build_chunks(target, persist=args.persist)

    sources = sorted({c.source_file for c in chunks})
    print(
        f"Loaded + chunked {len(chunks)} chunks from {len(sources)} "
        f"file(s): {', '.join(sources)}"
    )

    if args.dry_run:
        print(f"[dry-run] no embedding/storing. timings_ms={sw.as_dict()}")
        return 0

    # Full path: embed + store. Imports are local so --dry-run needs no extras.
    from rag.indexing.embeddings import get_embedding_client
    from rag.indexing.vector_store import VectorStore

    embedder = get_embedding_client(settings)
    store = VectorStore.from_settings(settings)
    with sw.time("embed"):
        vectors = embedder.embed_texts([c.text for c in chunks])
    with sw.time("store"):
        stored = store.add(chunks, vectors)

    print(
        f"Stored {stored} chunks in Chroma collection '{settings.chroma_collection}' "
        f"(total now {store.count()})."
    )
    print(
        f"embedding_tokens={embedder.total_tokens} "
        f"embedding_cost_usd={embedder.total_cost_usd:.6f} timings_ms={sw.as_dict()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
