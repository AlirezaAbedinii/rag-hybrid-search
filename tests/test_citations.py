"""Phase 3 tests for citation parsing + mapping (deterministic, no network)."""
from __future__ import annotations

from dataclasses import dataclass

from rag.generation.citations import (
    build_citations,
    has_unresolved,
    parse_citation_indices,
)


@dataclass
class FakeChunk:
    chunk_id: str
    metadata: dict


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
