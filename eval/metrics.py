"""Evaluation metrics: correctness, faithfulness, retrieval relevance, citation accuracy.

Two kinds of metric live here:

* **LLM-as-judge** — a second LLM call grades the system's answer. The judge is
  any object satisfying the ``Judge`` protocol (in practice a
  ``rag.generation.llm_client.ChatClient``), so tests inject a scripted fake.
  Metrics: **correctness** (answer vs the hand-written ``expected_answer``) and
  **faithfulness** (is every claim supported by the retrieved context?).
* **Deterministic** — no LLM involved. Metrics: **retrieval relevance** (was a
  golden ``supporting_source`` present in the top-k retrieved chunks?) and
  **citation accuracy** (share of citations the verification judge marked
  supported — computed by the pipeline at answer time, tallied here).

Each returns a :class:`MetricResult` with a normalized ``score`` in [0, 1] and a
boolean ``passed`` (judge metrics also carry the raw 1–5 rating).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# A rating at or above this (on the 1–5 judge scale) counts as a pass.
PASS_THRESHOLD = 4


class Judge(Protocol):
    """Something that can grade text. ``ChatClient`` satisfies this."""

    def complete(self, system: str, user: str) -> object: ...


@dataclass(frozen=True)
class MetricResult:
    """One metric's outcome for one answer."""

    name: str
    score: float  # normalized to [0, 1]
    passed: bool
    rating: int | None = None  # raw 1–5 judge rating (None for deterministic checks)
    detail: str = ""


# --- Judge prompts --------------------------------------------------------
CORRECTNESS_SYSTEM = (
    "You are a strict grader for a question-answering system. Compare the "
    "SYSTEM ANSWER to the REFERENCE ANSWER for the given QUESTION. Judge only "
    "whether the system answer is factually correct and complete relative to the "
    "reference — ignore wording, style, and extra detail that is still correct. "
    "Respond on a single line exactly as 'Rating: N' where N is 1-5:\n"
    "5 = fully correct and complete; 4 = correct, minor omission; "
    "3 = partially correct; 2 = mostly wrong; 1 = wrong or irrelevant.\n"
    "Then, optionally, a short 'Reason:' line."
)

FAITHFULNESS_SYSTEM = (
    "You are a strict grader checking for hallucination. Given the CONTEXT that "
    "was retrieved and the SYSTEM ANSWER, decide whether every factual claim in "
    "the answer is directly supported by the context. Outside knowledge that is "
    "not in the context counts as unsupported. Respond on a single line exactly "
    "as 'Rating: N' where N is 1-5:\n"
    "5 = every claim supported; 4 = supported with a trivial unsupported aside; "
    "3 = mix of supported and unsupported; 2 = mostly unsupported; "
    "1 = fabricated / contradicts the context.\n"
    "Then, optionally, a short 'Reason:' line."
)

_RATING_RE = re.compile(r"(?:rating|score)\s*[:=]?\s*([1-5])", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"\b([1-5])\b")


def parse_rating(text: str) -> int:
    """Extract a 1–5 rating from a judge response, robust to formatting.

    Prefers an explicit ``Rating: N``; falls back to the first standalone 1–5.
    Returns the worst rating (1) if nothing parseable is found, so an
    unparseable judge response fails closed rather than silently passing.
    """
    match = _RATING_RE.search(text) or _FALLBACK_RE.search(text)
    return int(match.group(1)) if match else 1


def _normalize(rating: int) -> float:
    """Map a 1–5 rating to [0, 1]."""
    return (rating - 1) / 4.0


def _judge_text(judge: Judge, system: str, user: str) -> str:
    """Call the judge and return its text, tolerating any ChatResult-like object."""
    result = judge.complete(system, user)
    return getattr(result, "text", str(result))


def build_correctness_prompt(question: str, expected: str, answer: str) -> tuple[str, str]:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{expected}\n\n"
        f"SYSTEM ANSWER:\n{answer}"
    )
    return CORRECTNESS_SYSTEM, user


def build_faithfulness_prompt(answer: str, context: str) -> tuple[str, str]:
    user = f"CONTEXT:\n{context}\n\nSYSTEM ANSWER:\n{answer}"
    return FAITHFULNESS_SYSTEM, user


def score_correctness(question: str, expected: str, answer: str, judge: Judge) -> MetricResult:
    """LLM-as-judge correctness of ``answer`` vs the hand-written ``expected``."""
    system, user = build_correctness_prompt(question, expected, answer)
    rating = parse_rating(_judge_text(judge, system, user))
    return MetricResult("correctness", _normalize(rating), rating >= PASS_THRESHOLD, rating)


def score_faithfulness(answer: str, context: str, judge: Judge) -> MetricResult:
    """LLM-as-judge faithfulness of ``answer`` to the retrieved ``context``."""
    system, user = build_faithfulness_prompt(answer, context)
    rating = parse_rating(_judge_text(judge, system, user))
    return MetricResult("faithfulness", _normalize(rating), rating >= PASS_THRESHOLD, rating)


def refusal_correctness(refused: bool) -> MetricResult:
    """Deterministic correctness for ``no_answer`` cases: passing == refused.

    For questions with no answer in the corpus, the *correct* behavior is to
    refuse. This needs no judge call — it is a direct check of the refusal flag.
    """
    return MetricResult(
        name="correctness",
        score=1.0 if refused else 0.0,
        passed=refused,
        rating=None,
        detail="refusal-based (no_answer category)",
    )


# --- Deterministic metrics (no judge) --------------------------------------
def _source_names(supporting_sources: list[str]) -> set[str]:
    """Golden sources normalized to bare filenames (``file#section`` -> ``file``)."""
    return {s.split("#", 1)[0].strip() for s in supporting_sources if s.strip()}


def score_retrieval_relevance(
    supporting_sources: list[str], contexts: list
) -> MetricResult | None:
    """Was any golden ``supporting_source`` among the retrieved chunks' sources?

    Deterministic top-k hit check (plan §6.2). Returns ``None`` for records with
    no supporting sources (``no_answer`` category — nothing to retrieve).
    Context objects need only a ``metadata`` dict with ``source_file``
    (``ScoredChunk`` satisfies this).
    """
    wanted = _source_names(supporting_sources)
    if not wanted:
        return None
    retrieved = {str(c.metadata.get("source_file", "")) for c in contexts}
    hit = bool(wanted & retrieved)
    return MetricResult(
        name="retrieval_relevance",
        score=1.0 if hit else 0.0,
        passed=hit,
        rating=None,
        detail=f"wanted one of {sorted(wanted)}; retrieved {sorted(retrieved)}",
    )


def score_citation_accuracy(citations: list) -> MetricResult | None:
    """Share of judged citations marked supported (from pipeline verification).

    Citation objects need ``resolved`` and ``supported`` attributes
    (``rag.generation.citations.Citation`` satisfies this). Unresolved
    citations count against accuracy — citing a chunk that was never retrieved
    is an accuracy failure, not a gap. Returns ``None`` when there are no
    citations or verification never ran (all ``supported`` are None).
    """
    if not citations:
        return None
    judged = [c for c in citations if not c.resolved or c.supported is not None]
    if not judged:
        return None
    good = sum(1 for c in judged if c.resolved and c.supported)
    score = good / len(judged)
    return MetricResult(
        name="citation_accuracy",
        score=score,
        passed=score >= 1.0,
        rating=None,
        detail=f"{good}/{len(judged)} citations supported",
    )
