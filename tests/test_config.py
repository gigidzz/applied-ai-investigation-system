"""Tests for the configuration layer.

Covers the four behaviours the rest of the system relies on: defaults are
correct, environment variables override them, invalid values are rejected at
construction rather than at use, and secrets never appear in output.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from ragsys.config import MissingSettingError, Settings, require


class IsolatedSettings(Settings):
    """Settings that ignore the developer's local .env file.

    Tests must not depend on local credentials. A suite that passes on one
    machine because of what happens to be in .env, and fails on another, is
    worse than no suite at all.

    Environment variables are still honoured, so a test can set one explicitly
    with monkeypatch.setenv and observe the effect.
    """

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


def test_defaults_are_sane() -> None:
    settings = IsolatedSettings()
    assert settings.corpus_dir.name == "corpus"
    assert settings.corpus_dir.is_absolute()
    assert settings.chunk_size == 160
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.embedding_dim == 384


def test_environment_variable_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "512")
    assert IsolatedSettings().chunk_size == 512


def test_overlap_not_smaller_than_chunk_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # An overlap equal to or larger than the chunk size means the chunker
    # never advances through the document.
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValidationError, match="chunk_overlap"):
        IsolatedSettings()


def test_value_outside_allowed_set_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        IsolatedSettings()


def test_secret_is_not_exposed_by_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    settings = IsolatedSettings()

    assert settings.llm_api_key is not None
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.llm_api_key)
    # Reachable only when asked for explicitly.
    assert settings.llm_api_key.get_secret_value() == "super-secret-value"


def test_require_names_the_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    import ragsys.config

    monkeypatch.setattr(ragsys.config.settings, "qdrant_url", None)
    with pytest.raises(MissingSettingError, match="QDRANT_URL"):
        require("qdrant_url")


def test_require_returns_the_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import ragsys.config

    monkeypatch.setattr(ragsys.config.settings, "qdrant_url", "http://localhost:6333")
    assert require("qdrant_url") == "http://localhost:6333"


def test_only_the_named_message_is_dropped():
    """Silencing a whole logger to hide one line also hides the useful ones."""
    import logging

    from ragsys.logging import DropMessagesContaining

    drop = DropMessagesContaining("unauthenticated requests")

    def record(message: str) -> logging.LogRecord:
        return logging.LogRecord("x", logging.WARNING, "f", 1, message, None, None)

    assert not drop.filter(record("You are sending unauthenticated requests to the Hub"))
    assert drop.filter(record("Retrying download after a 429 from the Hub"))
