"""The dense half of the index: text to a point in meaning-space.

The model is `BAAI/bge-small-en-v1.5` and it is executed by **ONNX Runtime**
rather than PyTorch. Same weights, same vectors — measured at 0.999999 cosine
similarity against the PyTorch output on both query and document text, which
is floating-point noise — but the process holds 345 MB instead of 626 MB.
PyTorch is a framework built for training on GPUs, and this system runs one
forward pass of a small model on a CPU; the training machinery was 489 MB of
the original footprint and none of it was used.

That difference decides where the system can be hosted. Several free tiers cap
a container at 512 MB, so the PyTorch build was excluded from them by 114 MB
and the ONNX build fits with room to spare.

The model is loaded lazily. Importing this module costs nothing, which keeps
the test suite fast and lets a web process decide for itself when to pay the
one-off cost of loading the weights.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ragsys.config import settings
from ragsys.logging import get_logger

if TYPE_CHECKING:
    from fastembed import TextEmbedding

log = get_logger(__name__)

# bge models are trained asymmetrically: queries carry a short instruction and
# passages carry none. Applying the prefix to both, or to neither, measurably
# degrades retrieval, and does so silently.
#
# This exact string matters. fastembed offers a `query_embed` helper that
# applies a prefix of its own, and its output matched the vectors in this
# index at only 0.96 to 0.99 — close enough to look fine and far enough to
# rank differently. The prefix is therefore applied here and the plain
# `embed` path is used for everything.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DimensionMismatch(RuntimeError):
    """The loaded model does not produce the dimension the config promises."""


def _declared_dimension(model: TextEmbedding, name: str) -> int | None:
    """The model's output size, as the library reports it.

    Tolerant of shape because the library has described its catalogue as both
    dictionaries and objects across versions, and a dimension check that
    crashes is worse than one that declines to run.
    """
    for description in model.list_supported_models():
        reported = (
            description.get("model")
            if isinstance(description, dict)
            else getattr(description, "model", None)
        )
        if reported != name:
            continue
        return (
            description.get("dim")
            if isinstance(description, dict)
            else getattr(description, "dim", None)
        )
    return None


def _quiet_download_progress() -> None:
    """Suppress model-download progress bars before the library loads.

    They write to stderr on every run and cannot be turned off afterwards,
    because the setting is read at import. Harmless locally, but they would
    interleave with real output in a deployment log. Set rather than
    overwritten, so an operator who wants them can export the variable.
    """
    for name in ("HF_HUB_DISABLE_PROGRESS_BARS", "TRANSFORMERS_NO_ADVISORY_WARNINGS"):
        os.environ.setdefault(name, "1")


class Embedder:
    """Encodes text with the embedding model.

    `embed_documents` and `embed_query` are deliberately separate methods
    rather than one method with a flag, because the asymmetric prefix is easy
    to apply to the wrong side and a flag invites exactly that mistake.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        expected_dim: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.expected_dim = expected_dim or settings.embedding_dim
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            _quiet_download_progress()
            from fastembed import TextEmbedding

            log.info("loading embedding model %s", self.model_name)
            model = TextEmbedding(model_name=self.model_name)
            actual = _declared_dimension(model, self.model_name)
            # Checked against the model rather than trusted from config: a
            # mismatch would otherwise surface as the vector store rejecting
            # every write, with nothing pointing at the real cause.
            if actual is not None and actual != self.expected_dim:
                raise DimensionMismatch(
                    f"{self.model_name} produces {actual} dimensions, "
                    f"but embedding_dim is set to {self.expected_dim}"
                )
            self._model = model
        return self._model

    @property
    def dim(self) -> int:
        return self.expected_dim

    @property
    def is_loaded(self) -> bool:
        """Whether the weights are in memory yet.

        Exposed so a health check can report readiness without triggering the
        load it is asking about: touching `.model` would block long enough for
        a platform health probe to decide the container is dead.
        """
        return self._model is not None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode passages for storage. No instruction prefix."""
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        """Encode a question for searching. Instruction prefix applied."""
        return next(iter(self.model.embed([QUERY_PREFIX + text]))).tolist()
