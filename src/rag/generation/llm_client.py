"""Single-provider chat client (OpenAI OR Anthropic). No multi-provider routing.

One provider is selected by ``settings.llm_provider``; there is deliberately no
runtime routing between providers. Both backends expose the same tiny interface —
``complete(system, user) -> ChatResult`` — so the pipeline and tests are
provider-agnostic and the LLM is trivially mockable. Provider SDKs are imported
lazily and the API key is validated before the first call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Settings, get_settings
from ..observability.metrics import TokenUsage


@dataclass(frozen=True)
class ChatResult:
    """A completion plus the token usage that produced it."""

    text: str
    usage: TokenUsage


class ChatClient(Protocol):
    """Minimal chat interface. Implementations are interchangeable/mockable."""

    model: str

    def complete(self, system: str, user: str) -> ChatResult: ...


class OpenAIChatClient:
    """OpenAI chat-completions backend (e.g. ``gpt-4o-mini``)."""

    def __init__(
        self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    'OpenAI generation requires \'openai\'. Install: pip install -e ".[llm]"'
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str) -> ChatResult:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return ChatResult(text=text, usage=usage)


class AnthropicChatClient:
    """Anthropic Messages backend (e.g. a Claude Sonnet model)."""

    def __init__(
        self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    'Anthropic generation requires \'anthropic\'. Install: pip install -e ".[llm]"'
                ) from exc
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str) -> ChatResult:
        client = self._ensure_client()
        resp = client.messages.create(
            model=self.model,
            system=system,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "input_tokens", 0),
            completion_tokens=getattr(resp.usage, "output_tokens", 0),
        )
        return ChatResult(text=text, usage=usage)


def get_chat_client(settings: Settings | None = None) -> ChatClient:
    """Build the configured single-provider chat client, validating its key."""
    settings = settings or get_settings()
    settings.validate_required_keys()
    if settings.llm_provider == "openai":
        return OpenAIChatClient(model=settings.generation_model, api_key=settings.openai_api_key)
    if settings.llm_provider == "anthropic":
        return AnthropicChatClient(
            model=settings.generation_model, api_key=settings.anthropic_api_key
        )
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")
