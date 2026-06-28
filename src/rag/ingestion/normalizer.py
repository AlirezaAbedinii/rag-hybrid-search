"""Text normalizer: clean raw block text and persist normalized documents.

Normalization is intentionally conservative and deterministic — it must not
mangle code snippets or config keys in technical docs. It only:

* normalizes line endings to ``\\n``,
* strips trailing whitespace on each line,
* collapses 3+ consecutive blank lines down to one, and
* trims leading/trailing blank lines.

Normalized documents are persisted to ``data/processed`` as JSON so the corpus
can be re-indexed without re-loading/re-parsing the originals.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .loaders import RawBlock, RawDocument

_TRAILING_WS = re.compile(r"[ \t]+(?=\n)|[ \t]+$")
_EXTRA_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Clean a single text span deterministically (see module docstring)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _EXTRA_BLANKS.sub("\n\n", text)
    return text.strip("\n")


def normalize_document(doc: RawDocument) -> RawDocument:
    """Return a copy of ``doc`` with every block normalized; drop empty blocks."""
    blocks: list[RawBlock] = []
    for block in doc.blocks:
        cleaned = normalize_text(block.text)
        if cleaned:
            blocks.append(
                RawBlock(text=cleaned, section_heading=block.section_heading, page=block.page)
            )
    return RawDocument(source_file=doc.source_file, doc_type=doc.doc_type, blocks=blocks)


def persist_processed(doc: RawDocument, processed_dir: str | Path) -> Path:
    """Write a normalized document to ``processed_dir`` as JSON; return its path."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{Path(doc.source_file).stem}.json"
    out_path.write_text(json.dumps(asdict(doc), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
