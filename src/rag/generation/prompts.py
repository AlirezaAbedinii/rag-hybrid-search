"""Prompt builders: grounded-generation prompt and citation-judge prompt.

The grounded-generation prompt is the heart of the MVP answer path. It presents
the retrieved chunks as a **numbered** context list and instructs the model to:

* answer **only** from that context,
* cite every claim inline with ``[n]`` (matching the context numbers), and
* explicitly refuse ("I don't know …") when the context is insufficient — never
  fabricate.

The numbering here is the contract the citation parser relies on: ``[n]`` in the
answer maps to the n-th context block (1-indexed). The citation-judge prompt is a
V1 placeholder.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

# The exact text returned when retrieval confidence is too low to answer.
REFUSAL_MESSAGE = (
    "I don't know based on the provided documentation. "
    "The retrieved context did not contain enough information to answer this question."
)

GROUNDED_SYSTEM_PROMPT = (
    "You are a precise technical-documentation assistant. Answer the user's "
    "question using ONLY the numbered context passages provided. Follow these "
    "rules strictly:\n"
    "1. Use only facts stated in the context. Do not use outside knowledge.\n"
    "2. Cite every claim inline with bracketed numbers like [1] or [2] that refer "
    "to the context passage the claim comes from. Cite multiple as [1][3].\n"
    "3. If the context does not contain enough information to answer, reply "
    'exactly: "' + REFUSAL_MESSAGE + '" and nothing else.\n'
    "4. Be concise and do not repeat the context verbatim."
)


class SupportsContextFields(Protocol):
    """The chunk fields the prompt needs (ScoredChunk satisfies this)."""

    text: str
    metadata: dict


def _source_label(metadata: dict) -> str:
    """Human-readable 'source_file § heading' label from chunk metadata."""
    source = metadata.get("source_file", "unknown")
    heading = metadata.get("section_heading") or ""
    page = metadata.get("page", -1)
    label = str(source)
    if heading:
        label += f" § {heading}"
    if isinstance(page, int) and page > 0:
        label += f" (p.{page})"
    return label


def build_context_block(contexts: Sequence[SupportsContextFields]) -> str:
    """Render retrieved chunks as a numbered, source-labelled context list."""
    lines: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        lines.append(f"[{i}] ({_source_label(ctx.metadata)})\n{ctx.text}")
    return "\n\n".join(lines)


def build_grounded_prompt(
    question: str, contexts: Sequence[SupportsContextFields]
) -> tuple[str, str]:
    """Return ``(system, user)`` messages for grounded generation."""
    context_block = build_context_block(contexts)
    user = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with [n] citations."
    )
    return GROUNDED_SYSTEM_PROMPT, user


CITATION_JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. Given a CLAIM from a generated "
    "answer and the SOURCE passage it cites, decide whether the source "
    "directly supports the claim. Paraphrase is fine; the source must state "
    "the substance of the claim. Outside knowledge does not count.\n"
    "Reply on a single line with exactly one word: SUPPORTED or UNSUPPORTED. "
    "Then, optionally, a short 'Reason:' line."
)


def build_citation_judge_prompt(claim: str, source_text: str) -> tuple[str, str]:
    """Return ``(system, user)`` asking a judge whether ``source_text`` supports ``claim``."""
    user = f"CLAIM:\n{claim}\n\nSOURCE:\n{source_text}"
    return CITATION_JUDGE_SYSTEM, user
