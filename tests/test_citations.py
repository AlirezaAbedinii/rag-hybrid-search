"""Tests for citation parsing, mapping, and LLM-judge verification (judge mocked)."""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.generation.citations import (
    build_citations,
    citation_coverage,
    claim_for_citation,
    has_unresolved,
    parse_citation_indices,
    verify_citations,
)
from rag.generation.llm_client import ChatResult
from rag.observability.metrics import TokenUsage


@dataclass
class FakeChunk:
    chunk_id: str
    metadata: dict
    text: str = "chunk text"


def test_parse_simple_citations() -> None:
    assert parse_citation_indices("Ferry queues jobs [1] and retries them [2].") == [1, 2]


def test_parse_no_citations() -> None:
    assert parse_citation_indices("No citations at all here.") == []


def test_parse_repeated_citations_dedup_in_order() -> None:
    assert parse_citation_indices("See [2], then [1], then [2] again.") == [2, 1]


def test_parse_comma_group_and_adjacent() -> None:
    assert parse_citation_indices("Grouped [1, 2] and adjacent [3][4].") == [1, 2, 3, 4]


def test_parse_ignores_malformed_brackets() -> None:
    # [a], [], [1b] are not valid integer citations; [ 2 ] (spaces) is.
    assert parse_citation_indices("junk [a] [] [1b] but [ 2 ] is fine") == [2]


def test_build_citations_resolves_to_chunks() -> None:
    contexts = [
        FakeChunk("c1", {"source_file": "a.md", "section_heading": "Intro"}),
        FakeChunk("c2", {"source_file": "b.md", "section_heading": None}),
    ]
    citations = build_citations("Answer [1] and [2].", contexts)
    assert [(c.index, c.chunk_id, c.resolved) for c in citations] == [
        (1, "c1", True),
        (2, "c2", True),
    ]
    assert citations[0].source_file == "a.md"
    assert citations[0].section_heading == "Intro"
    assert not has_unresolved(citations)


def test_build_citations_flags_out_of_range() -> None:
    contexts = [FakeChunk("c1", {"source_file": "a.md"})]
    citations = build_citations("Claim [1] and bogus [5].", contexts)
    assert citations[0].resolved is True and citations[0].chunk_id == "c1"
    assert citations[1].resolved is False and citations[1].chunk_id == ""
    assert has_unresolved(citations)


def test_resolved_citations_only_reference_retrieved_chunks() -> None:
    # Acceptance: every resolved [n] maps to a real retrieved chunk_id.
    contexts = [FakeChunk("c1", {"source_file": "a.md"}), FakeChunk("c2", {"source_file": "b.md"})]
    valid_ids = {c.chunk_id for c in contexts}
    citations = build_citations("Mix [1] [2] [9].", contexts)
    for c in citations:
        if c.resolved:
            assert c.chunk_id in valid_ids


# --- verification (LLM judge, scripted) --------------------------------------
@dataclass
class ScriptedJudge:
    """Returns queued verdicts in order; records the prompts it graded."""

    verdicts: list[str]
    seen: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str) -> ChatResult:
        self.seen.append(user)
        return ChatResult(self.verdicts.pop(0), TokenUsage(30, 5))


def test_claim_for_citation_picks_the_sentences_with_the_marker() -> None:
    answer = "Jobs retry three times [1]. Failures land in the DLQ [2]. Grouped claim [1, 2]."
    claim = claim_for_citation(answer, 2)
    assert "DLQ" in claim
    assert "Grouped claim" in claim  # [1, 2] carries index 2 too
    assert "retry three times" not in claim


def test_verify_marks_supported_and_unsupported() -> None:
    contexts = [
        FakeChunk("c1", {"source_file": "a.md"}, text="Jobs retry three times."),
        FakeChunk("c2", {"source_file": "b.md"}, text="Completely unrelated text."),
    ]
    answer = "Jobs retry three times [1]. Ferry was founded in 1999 [2]."
    citations = build_citations(answer, contexts)
    judge = ScriptedJudge(["SUPPORTED", "UNSUPPORTED\nReason: not in source"])

    verified, usage = verify_citations(answer, citations, contexts, judge)

    assert verified[0].supported is True
    assert verified[1].supported is False  # the fabricated claim is flagged
    assert usage.total_tokens == 70  # 2 judge calls tallied
    # The judge saw claim + the right source text.
    assert "retry three times" in judge.seen[0]
    assert "Completely unrelated" in judge.seen[1]


def test_verify_skips_unresolved_and_fails_closed_on_garbage() -> None:
    contexts = [FakeChunk("c1", {"source_file": "a.md"}, text="alpha")]
    answer = "Claim [1]. Bogus [7]."
    citations = build_citations(answer, contexts)
    judge = ScriptedJudge(["no verdict words here at all"])

    verified, _ = verify_citations(answer, citations, contexts, judge)

    assert verified[0].supported is False  # unparseable -> fails closed
    assert verified[1].resolved is False and verified[1].supported is None  # never judged
    assert len(judge.seen) == 1  # only the resolved citation cost a call


def test_citation_coverage() -> None:
    contexts = [
        FakeChunk("c1", {"source_file": "a.md"}, text="x"),
        FakeChunk("c2", {"source_file": "b.md"}, text="y"),
    ]
    answer = "One [1]. Two [2]."
    citations = build_citations(answer, contexts)
    assert citation_coverage(citations) is None  # nothing verified yet

    verified, _ = verify_citations(
        answer, citations, contexts, ScriptedJudge(["SUPPORTED", "UNSUPPORTED"])
    )
    assert citation_coverage(verified) == 0.5
