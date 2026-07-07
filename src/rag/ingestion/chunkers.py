"""Chunkers: fixed | recursive | semantic, switchable via config.

Three strategies behind one ``split_text`` interface:

* **fixed** — sliding character windows with overlap (the measured baseline).
* **recursive** — structure-aware: split on markdown headings, then paragraphs,
  then sentences, only hard-cutting when a single sentence exceeds the budget.
* **semantic** — embed sentences and break where adjacent-sentence cosine
  similarity drops below a threshold (topic shift), then merge to the size
  budget. Needs an embedder (injected; the offline backend keeps it free).

Every chunk is tagged with the strategy that produced it and a **stable**
``chunk_id`` (a content hash) so re-ingesting identical input yields identical
ids — important for idempotent indexing and dedup. Fixed and recursive are
deterministic with no network I/O; semantic is deterministic given a
deterministic embedder (tests inject fakes).
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
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


_HEADING_SPLIT = re.compile(r"(?=^#{1,6}\s)", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _merge_to_budget(parts: list[str], chunk_size: int, joiner: str) -> list[str]:
    """Greedily merge consecutive parts into pieces no longer than chunk_size."""
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{joiner}{part}" if buffer else part
        if buffer and len(candidate) > chunk_size:
            merged.append(buffer)
            buffer = part
        else:
            buffer = candidate
    if buffer:
        merged.append(buffer)
    return merged


@dataclass
class RecursiveChunker:
    """Structure-aware splitter: headings -> paragraphs -> sentences -> hard cut.

    Each level splits only pieces that still exceed ``chunk_size``; sibling
    pieces are greedily re-merged up to the budget so chunks stay as large (and
    as coherent) as the structure allows. No overlap — structural boundaries
    replace it.
    """

    chunk_size: int = 800
    strategy: str = "recursive"

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        return self._split(text, level=0)

    def _split(self, text: str, level: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if level == 0:  # markdown headings
            parts = [p.strip() for p in _HEADING_SPLIT.split(text) if p.strip()]
            joiner = "\n"
        elif level == 1:  # paragraphs
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            joiner = "\n\n"
        elif level == 2:  # sentences
            parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
            joiner = " "
        else:  # hard character cut (a single oversized sentence)
            return [
                text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
            ]

        if len(parts) <= 1:  # this level's separator didn't help; go deeper
            return self._split(text, level + 1)

        pieces: list[str] = []
        for merged in _merge_to_budget(parts, self.chunk_size, joiner):
            if len(merged) <= self.chunk_size:
                pieces.append(merged)
            else:
                pieces.extend(self._split(merged, level + 1))
        return pieces


class SupportsEmbedTexts(Protocol):
    """The slice of EmbeddingClient the semantic chunker needs."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class SemanticChunker:
    """Embedding-based splitter: break where adjacent sentences change topic.

    Sentences are embedded (one batch per block); a breakpoint is placed
    wherever the cosine similarity between neighbours falls below
    ``breakpoint_threshold``. The resulting topic runs are merged up to
    ``chunk_size``; an oversized run falls back to a hard cut. Embedding at
    ingest time costs tokens unless the offline backend is configured.
    """

    embedder: SupportsEmbedTexts
    chunk_size: int = 800
    breakpoint_threshold: float = 0.5
    strategy: str = field(default="semantic")

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) <= 1:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        vectors = self.embedder.embed_texts(sentences)
        # Group sentences into topic runs, breaking on low adjacent similarity.
        runs: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            if _cosine(vectors[i - 1], vectors[i]) < self.breakpoint_threshold:
                runs.append([sentences[i]])
            else:
                runs[-1].append(sentences[i])

        pieces: list[str] = []
        for merged in _merge_to_budget([" ".join(r) for r in runs], self.chunk_size, "\n\n"):
            if len(merged) <= self.chunk_size:
                pieces.append(merged)
            else:
                pieces.extend(
                    merged[i : i + self.chunk_size] for i in range(0, len(merged), self.chunk_size)
                )
        return pieces


def get_chunker(
    settings: Settings | None = None, embedder: SupportsEmbedTexts | None = None
) -> Chunker:
    """Build the configured chunker.

    ``semantic`` needs an embedder: pass one explicitly (tests, offline swap) or
    the configured embedding client is constructed — which may require an API
    key, unlike the other strategies.
    """
    settings = settings or get_settings()
    strategy = settings.chunk_strategy
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=settings.chunk_size)
    if strategy == "semantic":
        if embedder is None:
            from ..indexing.embeddings import get_embedding_client

            embedder = get_embedding_client(settings)
        return SemanticChunker(
            embedder=embedder,
            chunk_size=settings.chunk_size,
            breakpoint_threshold=settings.semantic_breakpoint_threshold,
        )
    raise ValueError(f"Unknown chunk_strategy: {strategy!r}")


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
