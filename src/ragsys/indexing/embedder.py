"""The dense half of the index: text to a point in meaning-space.

The model is loaded lazily. Importing this module costs nothing, which keeps
the test suite fast and lets a web process decide for itself when to pay the
one-off cost of loading roughly 130 MB of weights.

Vectors are normalised to unit length at encode time. Cosine similarity
between unit vectors is just their dot product, so normalising once here saves
the vector store from dividing by magnitudes on every comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragsys.config import settings
from ragsys.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

# bge models are trained asymmetrically: queries carry a short instruction and
# passages carry none. Applying the prefix to both, or to neither, measurably
# degrades retrieval, and does so silently.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DimensionMismatch(RuntimeError):
    """The loaded model does not produce the dimension the config promises."""


class Embedder:
    """Encodes text with a sentence-transformer model.

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
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s", self.model_name)
            model = SentenceTransformer(self.model_name, device="cpu")
            # Renamed upstream; the old name still works but warns.
            reader = getattr(model, "get_embedding_dimension", None) or (
                model.get_sentence_embedding_dimension
            )
            actual = reader()
            # Checked against the model rather than trusted from config: a
            # mismatch here would otherwise surface as the vector store
            # rejecting every write, with no indication of the real cause.
            if actual != self.expected_dim:
                raise DimensionMismatch(
                    f"{self.model_name} produces {actual} dimensions, "
                    f"but embedding_dim is set to {self.expected_dim}"
                )
            self._model = model
        return self._model

    @property
    def dim(self) -> int:
        return self.expected_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode passages for storage. No instruction prefix."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Encode a question for searching. Instruction prefix applied."""
        vector = self.model.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()
