"""Where chunks go after ingestion.

The pipeline talks to a ChunkStore and nothing else, so the same ingestion
code serves two very different destinations: a JSONL file during the offline
bulk build, and a vector database in Stage 3. Writing the protocol now is what
makes that swap a one-line change rather than a rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Protocol

from ragsys.ingestion.models import Chunk


class ChunkStore(Protocol):
    """The only interface the ingestion pipeline needs."""

    def upsert(self, chunks: list[Chunk]) -> int:
        """Add or replace chunks. Returns how many were written."""
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to a document. Returns how many."""
        ...

    def count(self) -> int: ...


class InMemoryChunkStore:
    """A store that keeps everything in a dictionary.

    Used by the tests and by anything that needs real add/update/delete
    behaviour without a database. Deletion is exact, which is what lets the
    change-detection logic be verified before Stage 3 exists.
    """

    def __init__(self) -> None:
        self._by_doc: dict[str, list[Chunk]] = {}

    def upsert(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self._by_doc.setdefault(chunk.doc_id, [])
            existing = self._by_doc[chunk.doc_id]
            replaced = False
            for position, present in enumerate(existing):
                if present.chunk_id == chunk.chunk_id:
                    existing[position] = chunk
                    replaced = True
                    break
            if not replaced:
                existing.append(chunk)
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        removed = self._by_doc.pop(doc_id, [])
        return len(removed)

    def count(self) -> int:
        return sum(len(v) for v in self._by_doc.values())

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return list(self._by_doc.get(doc_id, []))

    def all_chunks(self) -> list[Chunk]:
        return [c for chunks in self._by_doc.values() for c in chunks]


class JsonlChunkStore:
    """Appends chunks to a JSONL file, one JSON object per line.

    Append-only by design: the bulk build always starts from an empty file, so
    there is nothing to delete, and streaming to disk keeps memory flat over
    67,000 documents. Deletion needs an indexed backend, which arrives with the
    vector store; calling it here fails loudly rather than silently doing
    nothing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._written = 0

    def upsert(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self._handle.write(
                json.dumps(chunk.to_record(), ensure_ascii=False, default=str) + "\n"
            )
        self._written += len(chunks)
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        raise NotImplementedError(
            "JsonlChunkStore is append-only; deletion needs an indexed store. "
            "Use InMemoryChunkStore for update and delete, or the vector store."
        )

    def count(self) -> int:
        return self._written

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlChunkStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
