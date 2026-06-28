"""Document loaders: md / txt / pdf / html -> raw text + metadata.

MVP: Markdown, text, PDF. V1: HTML. Each loader returns raw text plus
metadata {source_file, section_heading, page}. Deterministic, no network."""
