"""The keyword half of the index: text to weighted terms.

This is BM25, implemented directly rather than called from a library, because
it has to be split across two places: the document side of the score is baked
into the stored vector at index time, and the query side is applied at search
time. The vector store only computes a dot product, so

    stored_value(term) * query_value(term)

summed over shared terms has to equal BM25. Putting the term-frequency and
length-normalisation factors on the document side and the inverse document
frequency on the query side achieves exactly that.

Tokenisation is the other reason this is hand-written. A general tokeniser
splits `pay-gw-4.2.1` into fragments and loses the identifier, which is
precisely the thing dense retrieval is already bad at. Here a compound token
is kept whole *and* emitted as its parts, so a query for "payment gateway"
matches a document saying "payment-gateway" without the exact release number
dissolving into digits.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Words, numbers, and identifiers held together by dots or hyphens.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_SEPARATORS = re.compile(r"[.\-]")

# Standard BM25 constants. k1 controls how fast term frequency saturates; b
# controls how strongly long documents are penalised. Chunks are already
# length-capped at index time, so b matters less here than it would over whole
# documents, but it is kept at the usual value rather than tuned blind.
K1 = 1.2
B = 0.75


def tokenise(text: str) -> list[str]:
    """Lowercase tokens, with compound identifiers kept and also split.

    `inc-2026-03-14` yields the whole identifier plus `inc`, `2026`, `03` and
    `14`. The fragments are nearly worthless on their own, which is fine: they
    appear in many documents, so inverse document frequency drives their
    weight close to zero without needing a stop-word list.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        if "-" in token or "." in token:
            tokens.extend(part for part in _SEPARATORS.split(token) if part)
    return tokens


@dataclass
class SparseVector:
    """Qdrant's sparse representation: parallel index and value arrays."""

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


class SparseStats:
    """Corpus statistics BM25 needs, gathered in one pass over the chunks.

    Held separately from the encoder because they must be identical at index
    time and at query time. Statistics computed over a different corpus than
    the one indexed would weight terms inconsistently, so they are written to
    disk alongside the index and loaded rather than recomputed.
    """

    def __init__(
        self,
        document_frequency: dict[str, int],
        total_documents: int,
        average_length: float,
    ) -> None:
        self.document_frequency = document_frequency
        self.total_documents = total_documents
        self.average_length = average_length or 1.0
        # Sorted so the token-to-index mapping is reproducible across builds.
        self.vocabulary = {
            token: position for position, token in enumerate(sorted(document_frequency))
        }

    def idf(self, token: str) -> float:
        """Inverse document frequency, in the BM25-with-floor form.

        The +1 inside the logarithm keeps the result positive for a term that
        appears in more than half the corpus, which the textbook form makes
        negative — and a negative weight would let a common word reduce a
        document's score below one that omits it entirely.
        """
        df = self.document_frequency.get(token)
        if not df:
            return 0.0
        return math.log(1.0 + (self.total_documents - df + 0.5) / (df + 0.5))

    def to_dict(self) -> dict:
        return {
            "total_documents": self.total_documents,
            "average_length": self.average_length,
            "document_frequency": self.document_frequency,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SparseStats:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            document_frequency=raw["document_frequency"],
            total_documents=raw["total_documents"],
            average_length=raw["average_length"],
        )

    @classmethod
    def from_texts(cls, texts: list[str]) -> SparseStats:
        """Convenience for tests and small corpora; the bulk build streams."""
        builder = StatsBuilder()
        for text in texts:
            builder.add(text)
        return builder.build()


class StatsBuilder:
    """Accumulates document frequencies without holding the corpus in memory."""

    def __init__(self) -> None:
        self._document_frequency: Counter[str] = Counter()
        self._documents = 0
        self._total_length = 0

    def add(self, text: str) -> None:
        tokens = tokenise(text)
        self._documents += 1
        self._total_length += len(tokens)
        # set(), not the full list: document frequency counts documents
        # containing the term, not occurrences of it.
        self._document_frequency.update(set(tokens))

    def build(self) -> SparseStats:
        average = self._total_length / self._documents if self._documents else 1.0
        return SparseStats(
            document_frequency=dict(self._document_frequency),
            total_documents=self._documents,
            average_length=average,
        )


class SparseEncoder:
    """Produces the stored and the query side of a BM25 score."""

    def __init__(self, stats: SparseStats, k1: float = K1, b: float = B) -> None:
        self.stats = stats
        self.k1 = k1
        self.b = b

    def encode_document(self, text: str) -> SparseVector:
        """Term frequency with length normalisation. No idf: that is the query's.

        A term absent from the vocabulary is dropped, because the vector store
        addresses dimensions by integer and an unknown token has no integer.
        It can only be absent if the statistics were built over a different
        corpus, which the index build prevents by construction.
        """
        tokens = tokenise(text)
        if not tokens:
            return SparseVector([], [])
        length = len(tokens)
        normaliser = self.k1 * (1.0 - self.b + self.b * length / self.stats.average_length)

        indices: list[int] = []
        values: list[float] = []
        for token, frequency in Counter(tokens).items():
            position = self.stats.vocabulary.get(token)
            if position is None:
                continue
            indices.append(position)
            values.append(frequency * (self.k1 + 1.0) / (frequency + normaliser))
        return SparseVector(indices, values)

    def encode_query(self, text: str) -> SparseVector:
        """Inverse document frequency per query term, one entry per term."""
        indices: list[int] = []
        values: list[float] = []
        for token in dict.fromkeys(tokenise(text)):
            position = self.stats.vocabulary.get(token)
            if position is None:
                continue
            weight = self.stats.idf(token)
            if weight <= 0.0:
                continue
            indices.append(position)
            values.append(weight)
        return SparseVector(indices, values)
