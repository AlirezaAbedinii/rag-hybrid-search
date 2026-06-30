"""Citation parsing (MVP) + verification (V1).

The MVP parser extracts inline ``[n]`` references from a generated answer and
maps each to the retrieved context chunk it points at (1-indexed, matching the
numbered context the prompt built). It is deterministic and handles the messy
cases models produce: no citations, repeated citations, comma groups (``[1, 2]``),
adjacent citations (``[1][2]``), and malformed brackets (``[a]``, ``[]``).

A parsed index that falls outside the retrieved set is kept but flagged
``resolved=False`` so the pipeline can surface an out-of-range/unsupported
citation rather than silently dropping it. LLM-as-judge *verification* of whether
a source actually supports a claim is a V1 feature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# Matches a bracket holding one or more comma-separated integers, e.g.
# [1], [1,2], [ 1 , 2 ]. Non-numeric brackets like [a] or [] do not match.
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


class SupportsChunkIdentity(Protocol):
    """The chunk fields a citation needs (ScoredChunk satisfies this)."""

    chunk_id: str
    metadata: dict


@dataclass(frozen=True)
class Citation:
    """A parsed ``[n]`` reference mapped to its source chunk (if any)."""

    index: int  # the 1-based number as written in the answer
    resolved: bool
    chunk_id: str = ""
    source_file: str = ""
    section_heading: str | None = None


def parse_citation_indices(text: str) -> list[int]:
    """Return the ordered, de-duplicated ``[n]`` indices appearing in ``text``.

    Order follows first appearance. ``[1, 2]`` yields ``[1, 2]``; repeats collapse.
    """
    seen: dict[int, None] = {}
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            seen.setdefault(int(part.strip()), None)
    return list(seen)


def build_citations(text: str, contexts: list[SupportsChunkIdentity]) -> list[Citation]:
    """Parse citations from ``text`` and resolve them against ``contexts``.

    A citation ``[n]`` resolves to ``contexts[n - 1]`` when ``1 <= n <= len``;
    otherwise it is returned with ``resolved=False`` and empty source fields.
    """
    citations: list[Citation] = []
    for index in parse_citation_indices(text):
        if 1 <= index <= len(contexts):
            ctx = contexts[index - 1]
            citations.append(
                Citation(
                    index=index,
                    resolved=True,
                    chunk_id=ctx.chunk_id,
                    source_file=str(ctx.metadata.get("source_file", "")),
                    section_heading=(ctx.metadata.get("section_heading") or None),
                )
            )
        else:
            citations.append(Citation(index=index, resolved=False))
    return citations


def has_unresolved(citations: list[Citation]) -> bool:
    """True if any citation points outside the retrieved context set."""
    return any(not c.resolved for c in citations)
