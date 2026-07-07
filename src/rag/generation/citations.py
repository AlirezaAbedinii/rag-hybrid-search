"""Citation parsing + LLM-as-judge verification.

**Parsing** extracts inline ``[n]`` references from a generated answer and maps
each to the retrieved context chunk it points at (1-indexed, matching the
numbered context the prompt built). It is deterministic and handles the messy
cases models produce: no citations, repeated citations, comma groups (``[1, 2]``),
adjacent citations (``[1][2]``), and malformed brackets (``[a]``, ``[]``).
A parsed index outside the retrieved set is kept but flagged ``resolved=False``.

**Verification** re-checks each *resolved* citation with an LLM judge: the
claim (the answer sentences carrying that ``[n]``) is shown next to the cited
chunk, and the judge answers SUPPORTED/UNSUPPORTED. An unparseable judge
response **fails closed** to unsupported — a citation is never silently trusted.
The judge is any ``ChatClient``, so tests script it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol

from ..observability.metrics import TokenUsage
from .prompts import build_citation_judge_prompt

# Matches a bracket holding one or more comma-separated integers, e.g.
# [1], [1,2], [ 1 , 2 ]. Non-numeric brackets like [a] or [] do not match.
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


class SupportsChunkIdentity(Protocol):
    """The chunk fields a citation needs (ScoredChunk satisfies this)."""

    chunk_id: str
    metadata: dict


@dataclass(frozen=True)
class Citation:
    """A parsed ``[n]`` reference mapped to its source chunk (if any).

    ``supported`` is set by verification: True/False once judged, None when
    verification hasn't run (disabled, or the citation never resolved).
    """

    index: int  # the 1-based number as written in the answer
    resolved: bool
    chunk_id: str = ""
    source_file: str = ""
    section_heading: str | None = None
    supported: bool | None = None


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


# --- Verification (LLM-as-judge) -------------------------------------------
class SupportsComplete(Protocol):
    """A chat client used as the verification judge (ChatClient satisfies it)."""

    def complete(self, system: str, user: str) -> object: ...


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_UNSUPPORTED_RE = re.compile(r"\bunsupported\b", re.IGNORECASE)
_SUPPORTED_RE = re.compile(r"\bsupported\b", re.IGNORECASE)


def claim_for_citation(answer: str, index: int) -> str:
    """The answer sentences carrying ``[index]`` — the claim that citation backs."""
    marker = re.compile(rf"\[\s*(?:\d+\s*,\s*)*{index}(?:\s*,\s*\d+)*\s*\]")
    sentences = [s for s in _SENTENCE_SPLIT.split(answer) if marker.search(s)]
    return " ".join(sentences).strip() or answer.strip()


def _parse_verdict(text: str) -> bool:
    """True only on an explicit SUPPORTED verdict; anything else fails closed."""
    if _UNSUPPORTED_RE.search(text):
        return False
    return bool(_SUPPORTED_RE.search(text))


def verify_citations(
    answer: str,
    citations: list[Citation],
    contexts: list[SupportsChunkIdentity],
    judge: SupportsComplete,
) -> tuple[list[Citation], TokenUsage]:
    """Judge every resolved citation; return updated citations + judge token usage.

    Unresolved citations keep ``supported=None`` (there is nothing to check
    against — ``resolved=False`` already flags them).
    """
    verified: list[Citation] = []
    prompt_tokens = completion_tokens = 0
    for citation in citations:
        if not citation.resolved:
            verified.append(citation)
            continue
        context = contexts[citation.index - 1]
        claim = claim_for_citation(answer, citation.index)
        system, user = build_citation_judge_prompt(claim, getattr(context, "text", ""))
        result = judge.complete(system, user)
        usage = getattr(result, "usage", None)
        if usage is not None:
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
        verdict = _parse_verdict(getattr(result, "text", str(result)))
        verified.append(replace(citation, supported=verdict))
    return verified, TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def citation_coverage(citations: list[Citation]) -> float | None:
    """Share of citations verified as supported; None when nothing was verified."""
    judged = [c for c in citations if c.supported is not None]
    if not judged:
        return None
    return sum(1 for c in judged if c.supported) / len(judged)
