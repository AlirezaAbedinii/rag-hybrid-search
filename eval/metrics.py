"""Evaluation metrics: correctness, faithfulness (MVP), retrieval relevance, citation accuracy (V1).

These are **LLM-as-judge** metrics: a second LLM call grades the system's answer.
The judge is any object satisfying the ``Judge`` protocol — in practice a
``rag.generation.llm_client.ChatClient`` — so tests inject a scripted fake and run
deterministically with no network.

MVP metrics implemented here:

* **correctness** — does the answer match the hand-written ``expected_answer``?
* **faithfulness** — is every claim in the answer supported by the retrieved
  context (i.e. no hallucination)?

Each returns a :class:`MetricResult` with a normalized ``score`` in [0, 1], a
boolean ``passed``, and the raw 1–5 judge rating. Retrieval-relevance and
citation-accuracy metrics are V1.
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
