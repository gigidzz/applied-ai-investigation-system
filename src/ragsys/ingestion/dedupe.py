"""Deciding whether a document is new, unchanged, modified, or a duplicate.

Ingestion is not a one-shot import. The same corpus gets rebuilt, a document
gets edited and re-uploaded, two different files turn out to hold identical
text. Four cases, and they need different handling:

    NEW        never seen. Chunk it and store it.
    UNCHANGED  same doc_id, same content hash. Skip entirely — re-chunking and
               re-embedding identical text wastes the most expensive step in
               the pipeline.
    CHANGED    same doc_id, different content hash. Remove every chunk the old
               version produced, then store the new version's chunks. Without
               the removal, stale chunks stay searchable forever and the system
               cites text that no longer exists.
    DUPLICATE  different doc_id, identical content hash. Skip, and record which
               document owns the text, so a later citation can point at the
               original.

The DUPLICATE case is the one Stage 1's title-based deduplication could not
catch: profiling found 18 distinct texts published under more than one title.
Identity by content hash catches them; identity by name cannot.

This index lives in memory. Stage 3 rebuilds it from the vector store on
startup, so the same four decisions apply against a persistent backend.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ragsys.ingestion.models import RawDocument


class DocumentState(StrEnum):
    NEW = "new"
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class Decision:
    """What to do with one document, and why."""

    doc_id: str
    state: DocumentState
    content_hash: str
    duplicate_of: str | None = None

    @property
    def should_store(self) -> bool:
        return self.state in (DocumentState.NEW, DocumentState.CHANGED)

    @property
    def should_remove_old(self) -> bool:
        return self.state is DocumentState.CHANGED


class ContentIndex:
    """Remembers what has been ingested, by identity and by content."""

    def __init__(self) -> None:
        self._hash_by_doc: dict[str, str] = {}
        self._doc_by_hash: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._hash_by_doc)

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, str]]) -> ContentIndex:
        """Rebuild from (doc_id, content_hash) pairs, as Stage 3 will from Qdrant."""
        index = cls()
        for doc_id, digest in records:
            index._hash_by_doc[doc_id] = digest
            index._doc_by_hash.setdefault(digest, doc_id)
        return index

    def classify(self, doc: RawDocument) -> Decision:
        """Decide this document's state without modifying the index."""
        digest = doc.hash
        known = self._hash_by_doc.get(doc.doc_id)

        if known == digest:
            return Decision(doc.doc_id, DocumentState.UNCHANGED, digest)
        if known is not None:
            return Decision(doc.doc_id, DocumentState.CHANGED, digest)

        owner = self._doc_by_hash.get(digest)
        if owner is not None and owner != doc.doc_id:
            return Decision(doc.doc_id, DocumentState.DUPLICATE, digest, duplicate_of=owner)

        return Decision(doc.doc_id, DocumentState.NEW, digest)

    def record(self, doc: RawDocument) -> None:
        """Remember a document we have just stored."""
        digest = doc.hash
        previous = self._hash_by_doc.get(doc.doc_id)
        if previous is not None and self._doc_by_hash.get(previous) == doc.doc_id:
            # This document owned its old hash; release it before reassigning.
            del self._doc_by_hash[previous]
        self._hash_by_doc[doc.doc_id] = digest
        self._doc_by_hash.setdefault(digest, doc.doc_id)

    def forget(self, doc_id: str) -> None:
        """Remove a document, as when it is deleted from the corpus."""
        digest = self._hash_by_doc.pop(doc_id, None)
        if digest is not None and self._doc_by_hash.get(digest) == doc_id:
            del self._doc_by_hash[digest]
