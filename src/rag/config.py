"""Central configuration for the RAG hybrid-search service.

Loads all settings from environment variables (and an optional ``.env`` file):
provider selection, API keys, model IDs, token prices, retrieval thresholds, and
filesystem paths. Importing this module never requires a secret to be set — keys
are validated lazily via :meth:`Settings.validate_required_keys` only when a code
path actually needs to call a provider. This keeps CI and ``make test`` green
without any secrets committed.

Every variable here is documented in ``.env.example``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three parents up from this file (src/rag/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when configuration is invalid (e.g. a required API key is missing)."""


LLMProvider = Literal["openai", "anthropic"]
EmbeddingProvider = Literal["openai", "sentence_transformers"]
ChunkStrategy = Literal["fixed", "recursive", "semantic"]
RetrievalMode = Literal["dense", "hybrid"]


class Settings(BaseSettings):
    """Application settings, populated from the environment.

    Field names are matched case-insensitively to environment variables, so the
    field ``top_k`` is set by the ``TOP_K`` env var. See ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- General -----------------------------------------------------------
    app_env: str = Field(default="dev", description="Deployment environment label.")
    log_level: str = Field(default="INFO", description="Python logging level.")

    # -- Provider selection ------------------------------------------------
    llm_provider: LLMProvider = Field(
        default="openai", description="Generation provider — pick ONE (no routing)."
    )
    embedding_provider: EmbeddingProvider = Field(
        default="openai",
        description="Embedding backend. 'sentence_transformers' runs offline/free.",
    )

    # -- API keys (validated lazily, not at import) ------------------------
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")

    # -- Model IDs ---------------------------------------------------------
    generation_model: str = Field(
        default="gpt-4o-mini",
        description="Generation model id (cheap tier in dev, strong tier for final eval).",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model id (or a sentence-transformers id for offline).",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Local cross-encoder reranker model id (V1).",
    )

    # -- Token prices (USD per 1,000,000 tokens) ---------------------------
    # Cost = tokens * price. Defaults reflect gpt-4o-mini + text-embedding-3-small.
    price_generation_input_per_1m: float = Field(
        default=0.15, description="USD per 1M generation prompt (input) tokens."
    )
    price_generation_output_per_1m: float = Field(
        default=0.60, description="USD per 1M generation completion (output) tokens."
    )
    price_embedding_per_1m: float = Field(
        default=0.02, description="USD per 1M embedding tokens."
    )

    # -- Retrieval / fusion / rerank ---------------------------------------
    default_mode: RetrievalMode = Field(
        default="dense", description="Default retrieval mode (MVP=dense, V1 adds hybrid)."
    )
    top_k: int = Field(default=10, ge=1, description="Top-k chunks retrieved per source.")
    rerank_candidates: int = Field(
        default=20, ge=1, description="Candidate pool size fed to the reranker (V1)."
    )
    rerank_top_k: int = Field(
        default=5, ge=1, description="Chunks kept after reranking (V1)."
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF rank constant (V1).")
    rrf_dense_weight: float = Field(
        default=0.7, ge=0.0, description="RRF weight on the dense ranking (V1)."
    )
    rrf_sparse_weight: float = Field(
        default=0.3, ge=0.0, description="RRF weight on the sparse (BM25) ranking (V1)."
    )

    # -- Chunking ----------------------------------------------------------
    chunk_strategy: ChunkStrategy = Field(
        default="fixed", description="Active chunking strategy (MVP=fixed)."
    )
    chunk_size: int = Field(default=800, ge=1, description="Target chunk size (chars/tokens).")
    chunk_overlap: int = Field(default=120, ge=0, description="Overlap between adjacent chunks.")

    # -- Thresholds --------------------------------------------------------
    dedup_cosine_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Skip chunks above this cosine sim (V1)."
    )
    retrieval_confidence_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Below this, return the 'I don't know' response instead of generating.",
    )
    confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Composite-confidence floor for surfacing an answer as high-confidence (V1).",
    )

    # -- Paths (resolved relative to the repo root if not absolute) --------
    data_raw_dir: Path = Field(default=Path("data/raw"), description="Raw documents root.")
    data_processed_dir: Path = Field(
        default=Path("data/processed"), description="Normalized documents root."
    )
    corpus_dir: Path = Field(
        default=Path("data/raw/ferry_docs"),
        description="Sample corpus to seed (provided ferry docs).",
    )
    chroma_persist_dir: Path = Field(
        default=Path("data/chroma"), description="ChromaDB persistent volume."
    )
    chroma_collection: str = Field(
        default="ferry_docs", description="Chroma collection name."
    )
    bm25_index_path: Path = Field(
        default=Path("data/bm25_index.pkl"), description="Persisted BM25 index (V1)."
    )
    trace_store_path: Path = Field(
        default=Path("data/traces.sqlite"), description="Per-request trace store."
    )
    golden_set_path: Path = Field(
        default=Path("eval/golden/golden_set.jsonl"),
        description="Hand-verified golden eval set.",
    )

    # -- Derived helpers ---------------------------------------------------
    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Make every relative path absolute against the repo root."""
        for name in (
            "data_raw_dir",
            "data_processed_dir",
            "corpus_dir",
            "chroma_persist_dir",
            "bm25_index_path",
            "trace_store_path",
            "golden_set_path",
        ):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                object.__setattr__(self, name, (REPO_ROOT / value).resolve())
        return self

    @property
    def active_api_key(self) -> str:
        """The API key for the selected generation provider ('' if unset)."""
        return self.openai_api_key if self.llm_provider == "openai" else self.anthropic_api_key

    def validate_required_keys(self) -> None:
        """Raise :class:`ConfigError` if a key needed by the active config is missing.

        Call this from code paths that actually hit a provider — not at import.
        The error message names the exact environment variable to set.
        """
        missing: list[str] = []
        if not self.active_api_key:
            var = "OPENAI_API_KEY" if self.llm_provider == "openai" else "ANTHROPIC_API_KEY"
            missing.append(var)
        # OpenAI embeddings need a key too; the offline backend does not.
        if self.embedding_provider == "openai" and not self.openai_api_key:
            if "OPENAI_API_KEY" not in missing:
                missing.append("OPENAI_API_KEY")
        if missing:
            raise ConfigError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". Set "
                + ("it" if len(missing) == 1 else "them")
                + " in your environment or .env file (see .env.example)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance loaded from the environment.

    Importing/calling this never requires a secret; use
    :meth:`Settings.validate_required_keys` before any provider call.
    """
    return Settings()
