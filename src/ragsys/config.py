"""Central configuration for the whole system.

Every setting lives here exactly once, with a type and a default. Values are
read from environment variables or the project's .env file, in that order of
precedence. Secrets are wrapped in SecretStr so they render as '**********'
if they ever reach a log line or a traceback.

Anywhere else in the codebase:

    from ragsys.config import settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/ragsys/config.py -> src/ragsys -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MissingSettingError(RuntimeError):
    """Raised when a setting is needed but was never provided."""


class Settings(BaseSettings):
    """All runtime configuration, validated at import time."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- filesystem ----------
    data_dir: Path = PROJECT_ROOT / "data"
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"

    # ---------- logging ----------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # ---------- document chunking ----------
    chunk_strategy: Literal["fixed", "recursive", "structural"] = "recursive"
    chunk_size: int = Field(default=300, gt=0, description="target chunk length in words")
    chunk_overlap: int = Field(default=50, ge=0, description="word overlap between chunks")

    # ---------- embeddings ----------
    # Changing the model invalidates every stored vector: the corpus must be
    # re-embedded, because vectors produced by different models are not
    # comparable to each other.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ---------- vector store ----------
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "ragsys_chunks"

    # ---------- retrieval ----------
    retrieval_top_k: int = 20
    rerank_top_k: int = 15

    # ---------- answer generation ----------
    llm_provider: Literal["gemini", "groq", "anthropic", "openai"] = "gemini"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    # ---------- answer evaluation ----------
    # The judge is deliberately a different model from the generator, so that a
    # model is never grading its own output. Self-preference bias inflates
    # correctness scores and makes the reported metrics untrustworthy.
    judge_provider: Literal["gemini", "groq", "anthropic", "openai"] | None = None
    judge_model: str | None = None
    judge_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking never advances."
            )
        return self


settings = Settings()


def require(name: str) -> object:
    """Return a setting's value, or fail immediately with a readable message.

    Call this where a credential is first needed, so that a missing key reports
    itself as "QDRANT_URL is not set" instead of letting a None travel deep
    into an unrelated stack trace.
    """
    value = getattr(settings, name, None)
    if isinstance(value, SecretStr):
        value = value.get_secret_value() or None
    if value is None or value == "":
        raise MissingSettingError(
            f"Setting '{name}' is required but not set. "
            f"Add {name.upper()} to your .env file (see .env.example)."
        )
    return value
