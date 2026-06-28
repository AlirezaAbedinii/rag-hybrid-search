"""Chunkers: fixed | recursive | semantic, switchable via config.

MVP: fixed-size-with-overlap (character windows). V1 adds structure-aware
recursive and semantic chunkers behind the same interface.

Every chunk is tagged with the strategy that produced it and a **stable**
``chunk_id`` (a content hash) so re-ingesting identical input yields identical
ids — important for idempotent indexing and dedup. Chunking is deterministic and
does no network I/O.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from ..config import Settings, get_settings
from .loaders import RawDocument


@dataclass(frozen=True)
class Chunk:
    """A unit of indexable text plus its provenance metadata."""

    chunk_id: str
    text: str
    source_file: str
    strategy: str
    ordinal: int
    section_heading: str | None = None
    page: int | None = None

    def metadata(self) -> dict[str, str | int]:
        """Chroma-safe metadata dict (no ``None`` values; Chroma rejects them)."""
        return {
            "source_file": self.source_file,
            "section_heading": self.section_heading or "",
            "page": self.page if self.page is not None else -1,
            "strategy": self.strategy,
            "ordinal": self.ordinal,
        }


def make_chunk_id(
    source_file: str,
    strategy: str,
    ordinal: int,
    section_heading: str | None,
    page: int | None,
    text: str,
) -> str:
    """Deterministic 16-hex-char id from a chunk's provenance + content."""
    page_part = page if page is not None else ""
    key = f"{source_file}|{strategy}|{ordinal}|{section_heading or ''}|{page_part}|{text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class Chunker(Protocol):
    """A chunker splits a block of text into ordered substrings."""

    strategy: str

    def split_text(self, text: str) -> list[str]: ...


@dataclass
class FixedSizeChunker:
    """Fixed-size sliding-window chunker with character overlap.

    ``chunk_size`` is the max window length; ``overlap`` characters are repeated
    between consecutive windows. The step is ``chunk_size - overlap`` (>= 1).
    """

    chunk_size: int = 800
    overlap: int = 120
    strategy: str = "fixed"

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    @property
    def step(self) -> int:
        return self.chunk_size - self.overlap

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        windows: list[str] = []
        start = 0
        while start < len(text):
            window = text[start : start + self.chunk_size]
            windows.append(window)
            if start + self.chunk_size >= len(text):
                break
            start += self.step
        return windows


def get_chunker(settings: Settings | None = None) -> Chunker:
    """Build the configured chunker. MVP supports ``fixed``; others are V1."""
    settings = settings or get_settings()
    strategy = settings.chunk_strategy
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    raise NotImplementedError(
        f"Chunking strategy '{strategy}' is a V1 feature; MVP supports 'fixed' only."
    )


def chunk_document(doc: RawDocument, chunker: Chunker) -> list[Chunk]:
    """Split a normalized document into ordered, metadata-tagged chunks."""
    chunks: list[Chunk] = []
    ordinal = 0
    for block in doc.blocks:
        for piece in chunker.split_text(block.text):
            chunk_id = make_chunk_id(
                doc.source_file, chunker.strategy, ordinal, block.section_heading, block.page, piece
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source_file=doc.source_file,
                    strategy=chunker.strategy,
                    ordinal=ordinal,
                    section_heading=block.section_heading,
                    page=block.page,
                )
            )
            ordinal += 1
    return chunks
