"""Phase 0 tests for src/rag/config.py.

Covers the Phase 0 acceptance criterion: settings load from the environment, and
a missing required key raises a clear, actionable error. All tests construct
``Settings`` with ``_env_file=None`` so a developer's local ``.env`` cannot make
the results non-deterministic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import REPO_ROOT, ConfigError, Settings, get_settings


def _settings(**env: str) -> Settings:
    """Build Settings from an explicit env dict, ignoring any on-disk .env."""
    return Settings(_env_file=None, **env)


def test_defaults_load_without_any_env() -> None:
    """Importing/constructing settings must not require secrets (CI-safe)."""
    s = _settings()
    assert s.llm_provider == "openai"
    assert s.embedding_provider == "openai"
    assert s.generation_model == "gpt-4o-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.default_mode == "dense"
    assert s.top_k == 10
    assert s.dedup_cosine_threshold == 0.95
    assert s.openai_api_key == ""  # no secret needed to construct


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Field values come from environment variables (case-insensitive)."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("GENERATION_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("RRF_DENSE_WEIGHT", "0.6")
    monkeypatch.setenv("PRICE_GENERATION_INPUT_PER_1M", "3.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    s = _settings()
    assert s.llm_provider == "anthropic"
    assert s.generation_model == "claude-sonnet-4-6"
    assert s.top_k == 7
    assert s.rrf_dense_weight == 0.6
    assert s.price_generation_input_per_1m == 3.0
    assert s.active_api_key == "sk-test"


def test_token_prices_and_thresholds_present() -> None:
    """Prices and thresholds the cost/SLO layer depends on are all loaded."""
    s = _settings()
    assert s.price_generation_input_per_1m > 0
    assert s.price_generation_output_per_1m > 0
    assert s.price_embedding_per_1m > 0
    assert 0.0 <= s.retrieval_confidence_threshold <= 1.0
    assert 0.0 <= s.confidence_threshold <= 1.0


def test_paths_resolved_to_absolute_under_repo_root() -> None:
    s = _settings()
    assert s.corpus_dir.is_absolute()
    assert s.golden_set_path.is_absolute()
    assert str(s.corpus_dir).startswith(str(REPO_ROOT))
    assert s.corpus_dir == (REPO_ROOT / "data/raw/ferry_docs").resolve()


def test_missing_required_key_raises_clear_error() -> None:
    """No key for the active provider -> ConfigError naming the exact env var."""
    s = _settings(openai_api_key="", anthropic_api_key="")
    with pytest.raises(ConfigError) as exc:
        s.validate_required_keys()
    msg = str(exc.value)
    assert "OPENAI_API_KEY" in msg
    assert ".env.example" in msg


def test_anthropic_provider_requires_anthropic_key() -> None:
    """Switching the generation provider changes which key is required."""
    s = _settings(
        llm_provider="anthropic",
        embedding_provider="sentence_transformers",
        anthropic_api_key="",
    )
    with pytest.raises(ConfigError) as exc:
        s.validate_required_keys()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_validate_passes_when_keys_present() -> None:
    s = _settings(llm_provider="openai", openai_api_key="sk-present")
    s.validate_required_keys()  # must not raise


def test_offline_embeddings_need_no_openai_key() -> None:
    """The offline embedding backend lifts the OpenAI key requirement."""
    s = _settings(
        llm_provider="anthropic",
        embedding_provider="sentence_transformers",
        anthropic_api_key="sk-a",
    )
    s.validate_required_keys()  # must not raise


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
    assert isinstance(get_settings().corpus_dir, Path)
