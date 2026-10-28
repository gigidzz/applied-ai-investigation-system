"""The two shapes every ingestion component speaks.

A loader produces RawDocument. A chunker turns one RawDocument into Chunks.
Nothing downstream of ingestion needs to know what a PDF is.

Identifiers are derived by hashing rather than counting, so that re-running
ingestion in a different order produces the same ids. That matters because
evaluation labels, and later the vector store's primary keys, reference them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Long enough that a collision across a few million records is implausible,
# short enough to stay readable in logs and citations.
_ID_LENGTH = 16


def content_hash(text: str) -> str:
    """Fingerprint of a piece of text, used to detect identical content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def stable_id(*parts: str) -> str:
    """Deterministic id from any number of string parts.

    The parts are joined with a separator that cannot appear in a path or an
    index, so ("a/b", "c") and ("a", "b/c") cannot collide.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:_ID_LENGTH]


@dataclass(frozen=True)
class RawDocument:
    """One independent unit of text extracted from a file.

    Most files yield exactly one of these. A CSV yields one per row, because
    each row is independently meaningful and independently citable: answering
    "what was deployed on 14 March?" should cite one row, not a 298-row file.
    """

    doc_id: str
    text: str
    source_path: str
    fmt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Chunk:
    """A retrievable, citable piece of a RawDocument.

    `index` is the position within the document, so citations can say which
    part of a long document a claim came from. `strategy` records how the cut
    was made, which is what lets the evaluation harness compare two chunking
    strategies over the same corpus without re-ingesting it.
    """

    chunk_id: str
    doc_id: str
    text: str
    index: int
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_record(self) -> dict[str, Any]:
        """Flat dict for JSONL output and, later, a vector-store payload."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "index": self.index,
            "strategy": self.strategy,
            "word_count": self.word_count,
            "content_hash": self.hash,
            **{f"meta_{k}": v for k, v in self.metadata.items()},
        }


def make_chunk(
    doc: RawDocument, text: str, index: int, strategy: str, extra: dict[str, Any] | None = None
) -> Chunk:
    """Build a Chunk, inheriting the document's metadata.

    The chunk inherits rather than references its document's metadata so that
    a retrieved chunk carries everything needed to cite it, with no second
    lookup at query time.
    """
    metadata = dict(doc.metadata)
    if extra:
        metadata.update(extra)
    metadata.setdefault("source_path", doc.source_path)
    metadata.setdefault("format", doc.fmt)
    return Chunk(
        chunk_id=stable_id(doc.doc_id, str(index), strategy),
        doc_id=doc.doc_id,
        text=text,
        index=index,
        strategy=strategy,
        metadata=metadata,
    )
