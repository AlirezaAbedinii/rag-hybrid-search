"""Ingestion: load, normalize, chunk, dedup raw documents.

This package exposes a small, deterministic, network-free pipeline that turns a
file or a directory of files into metadata-tagged :class:`Chunk` objects:

    load_path -> normalize_document -> chunk_document

Embedding + indexing (which need a provider/Chroma) live in ``rag.indexing`` and
are driven by ``scripts/ingest.py``; the helpers here are the testable core.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from .chunkers import Chunk, Chunker, chunk_document, get_chunker
from .loaders import RawDocument, is_supported, load_path
from .normalizer import normalize_document, persist_processed

__all__ = [
    "Chunk",
    "Chunker",
    "RawDocument",
    "build_chunks_for_file",
    "build_chunks_for_dir",
]


def build_chunks_for_file(
    path: str | Path,
    *,
    settings: Settings | None = None,
    chunker: Chunker | None = None,
    persist: bool = False,
) -> list[Chunk]:
    """Load, normalize, and chunk a single file into metadata-tagged chunks."""
    settings = settings or get_settings()
    chunker = chunker or get_chunker(settings)
    doc: RawDocument = normalize_document(load_path(path))
    if persist:
        persist_processed(doc, settings.data_processed_dir)
    return chunk_document(doc, chunker)


def build_chunks_for_dir(
    directory: str | Path,
    *,
    settings: Settings | None = None,
    persist: bool = False,
) -> list[Chunk]:
    """Recursively load every supported file in ``directory`` into chunks.

    Files are processed in sorted path order so the output is deterministic.
    Unsupported files (e.g. HTML until the V1 loader) are skipped.
    """
    settings = settings or get_settings()
    chunker = get_chunker(settings)
    directory = Path(directory)
    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and is_supported(path):
            chunks.extend(
                build_chunks_for_file(path, settings=settings, chunker=chunker, persist=persist)
            )
    return chunks
