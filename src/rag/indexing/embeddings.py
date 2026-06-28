"""Embedding client (provider-agnostic).

Default: OpenAI ``text-embedding-3-small``. Optional offline swap to a
``sentence-transformers`` model behind the **same interface** so the rest of the
system never knows which backend produced the vectors.

Every client tracks cumulative ``total_tokens`` and ``total_cost_usd`` so
ingestion can report embedding cost (instrumentation from day one). The offline
backend reports zero cost. Provider SDKs are imported lazily so importing this
module never requires them.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..observability.metrics import token_cost

Vector = list[float]


@runtime_checkable
class EmbeddingClient(Protocol):
    """Embeds text into dense vectors. Implementations are interchangeable."""

    model: str
    total_tokens: int
    total_cost_usd: float

    def embed_texts(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class OpenAIEmbeddingClient:
    """OpenAI embeddings backend (``text-embedding-3-small`` by default)."""

    def __init__(self, model: str, api_key: str, price_per_million: float) -> None:
        self.model = model
        self._api_key = api_key
        self._price_per_million = price_per_million
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._client = None  # lazily constructed OpenAI() client

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "OpenAI embeddings require 'openai'. Install: pip install -e \".[llm]\""
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        client = self._ensure_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        tokens = getattr(resp, "usage", None)
        if tokens is not None:
            self.total_tokens += tokens.total_tokens
            self.total_cost_usd += token_cost(tokens.total_tokens, self._price_per_million)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> Vector:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbeddingClient:
    """Offline embeddings backend (free; ``total_cost_usd`` stays 0)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._encoder = None

    def _ensure_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "Offline embeddings require 'sentence-transformers'. "
                    'Install: pip install -e ".[embeddings-offline]"'
                ) from exc
            self._encoder = SentenceTransformer(self.model)
        return self._encoder

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        encoder = self._ensure_encoder()
        vectors = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> Vector:
        return self.embed_texts([text])[0]


def get_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    """Build the configured embedding client, validating keys where needed."""
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        settings.validate_required_keys()
        return OpenAIEmbeddingClient(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            price_per_million=settings.price_embedding_per_1m,
        )
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddingClient(model=settings.embedding_model)
    raise ValueError(f"Unknown embedding_provider: {settings.embedding_provider!r}")
