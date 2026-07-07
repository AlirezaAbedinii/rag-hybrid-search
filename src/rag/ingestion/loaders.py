"""Document loaders: md / txt / pdf / html -> raw text + metadata.

Each loader returns a :class:`RawDocument` made of one or more :class:`RawBlock`
units. A *block* is a span of text that already carries the metadata we can know
at load time:

* Markdown -> one block per heading section (``section_heading`` set).
* Text     -> a single block (no heading, no page).
* PDF      -> one block per page (``page`` set, 1-indexed).
* HTML     -> one block per ``<h1>``-``<h6>`` section (``section_heading`` set);
  ``script``/``style``/``nav`` content is stripped.

Loaders are deterministic and do no network I/O. Heavy/optional parsers
(``pypdf``, ``bs4``) are imported lazily inside their loader so importing this
module stays cheap and dependency-free for the Markdown/text paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# File extensions each loader handles.
MARKDOWN_SUFFIXES = {".md", ".markdown"}
TEXT_SUFFIXES = {".txt", ".text", ""}
PDF_SUFFIXES = {".pdf"}
HTML_SUFFIXES = {".html", ".htm"}


class UnsupportedFormatError(ValueError):
    """Raised when a file's extension has no registered loader."""


@dataclass
class RawBlock:
    """A span of text plus the metadata known at load time."""

    text: str
    section_heading: str | None = None
    page: int | None = None


@dataclass
class RawDocument:
    """A loaded document: its source filename, type, and ordered blocks."""

    source_file: str
    doc_type: str
    blocks: list[RawBlock] = field(default_factory=list)


def load_text(path: Path) -> RawDocument:
    """Load a plain-text file as a single block."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return RawDocument(
        source_file=path.name,
        doc_type="text",
        blocks=[RawBlock(text=text)],
    )


def load_markdown(path: Path) -> RawDocument:
    """Load Markdown, splitting into one block per heading section.

    Uses a simple, deterministic ATX-heading (``#``) parser — no external
    dependency. Each block's ``section_heading`` is the nearest preceding
    heading (the document title for any preamble before the first heading).
    Fenced code blocks are not interpreted as headings.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    blocks: list[RawBlock] = []
    current_heading: str | None = None
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(RawBlock(text=text, section_heading=current_heading))
        buffer.clear()

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if not in_fence and stripped.startswith("#"):
            # Heading line: close the previous section, start a new one.
            flush()
            current_heading = stripped.lstrip("#").strip() or current_heading
            continue
        buffer.append(line)
    flush()

    if not blocks:  # document had a heading but no body text
        blocks.append(RawBlock(text="", section_heading=current_heading))
    return RawDocument(source_file=path.name, doc_type="markdown", blocks=blocks)


def load_pdf(path: Path) -> RawDocument:
    """Load a PDF as one block per page (``page`` 1-indexed).

    ``pypdf`` is imported lazily; install the ``ingestion`` extra to use this.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PDF loading requires 'pypdf'. Install the ingestion extra: "
            'pip install -e ".[ingestion]"'
        ) from exc

    path = Path(path)
    reader = PdfReader(str(path))
    blocks: list[RawBlock] = []
    for i, page in enumerate(reader.pages, start=1):
        blocks.append(RawBlock(text=page.extract_text() or "", page=i))
    return RawDocument(source_file=path.name, doc_type="pdf", blocks=blocks)


def load_html(path: Path) -> RawDocument:
    """Load HTML as one block per ``<h1>``-``<h6>`` section.

    Mirrors the markdown loader: each block's ``section_heading`` is the nearest
    preceding heading (any pre-heading content gets the document ``<title>`` if
    present). ``script``, ``style``, and ``nav`` elements are stripped.
    ``bs4`` is imported lazily; install the ``ingestion`` extra to use this.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "HTML loading requires 'beautifulsoup4'. Install the ingestion extra: "
            'pip install -e ".[ingestion]"'
        ) from exc

    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None
    body = soup.body or soup

    blocks: list[RawBlock] = []
    current_heading: str | None = title
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(RawBlock(text=text, section_heading=current_heading))
        buffer.clear()

    heading_names = {"h1", "h2", "h3", "h4", "h5", "h6"}
    for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        if element.name in heading_names:
            flush()
            current_heading = element.get_text(" ", strip=True) or current_heading
            continue
        text = element.get_text(" ", strip=True)
        if text:
            buffer.append(text)
    flush()

    if not blocks:  # no recognizable content elements — fall back to full text
        blocks.append(RawBlock(text=body.get_text(" ", strip=True), section_heading=title))
    return RawDocument(source_file=path.name, doc_type="html", blocks=blocks)


# Dispatch table: suffix set -> loader.
_LOADERS = [
    (MARKDOWN_SUFFIXES, load_markdown),
    (TEXT_SUFFIXES, load_text),
    (PDF_SUFFIXES, load_pdf),
    (HTML_SUFFIXES, load_html),
]


def load_path(path: str | Path) -> RawDocument:
    """Load a single file, dispatching on its extension.

    Raises :class:`UnsupportedFormatError` for unhandled extensions (e.g. HTML
    until the V1 loader lands).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    for suffixes, loader in _LOADERS:
        if suffix in suffixes:
            return loader(path)
    raise UnsupportedFormatError(
        f"No loader for '{suffix or path.name}'. Supported: markdown, text, pdf, html."
    )


def is_supported(path: str | Path) -> bool:
    """Return True if :func:`load_path` can handle this file's extension."""
    suffix = Path(path).suffix.lower()
    return any(suffix in suffixes for suffixes, _ in _LOADERS)
