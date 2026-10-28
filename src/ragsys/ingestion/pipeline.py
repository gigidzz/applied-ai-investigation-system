"""The one way documents enter the system.

Every path into the corpus goes through ingest_file: the offline bulk build
over 67,000 records, and the single-document upload in the web interface.
Two separate code paths would drift, and uploaded documents would end up
chunked or deduplicated differently from pre-loaded ones — a difference that
is invisible until someone's uploaded PDF behaves oddly.

    load  ->  enrich  ->  classify  ->  chunk  ->  store
    │         │           │             │          │
    │         │           │             │          └─ ChunkStore (file, or
    │         │           │             │             later a vector database)
    │         │           │             └─ one or both strategies
    │         │           └─ new / unchanged / changed / duplicate
    │         └─ extract date, author, service, references
    └─ format-specific reader
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ragsys.ingestion.chunking import STRATEGIES, chunk_document
from ragsys.ingestion.dedupe import ContentIndex, DocumentState
from ragsys.ingestion.loaders import SUPPORTED_SUFFIXES, UnsupportedFormatError, load_any
from ragsys.ingestion.metadata import enrich
from ragsys.ingestion.models import RawDocument
from ragsys.ingestion.stores import ChunkStore
from ragsys.logging import get_logger

log = get_logger(__name__)

DEFAULT_STRATEGIES: tuple[str, ...] = STRATEGIES


@dataclass(frozen=True)
class IngestResult:
    """What happened to one document."""

    doc_id: str
    state: DocumentState
    fmt: str
    chunks_written: int = 0
    chunks_removed: int = 0
    duplicate_of: str | None = None


@dataclass
class IngestSummary:
    """Totals across a batch, for the manifest and the logs."""

    files_seen: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    documents: int = 0
    chunks_written: int = 0
    chunks_removed: int = 0
    by_state: Counter[str] = field(default_factory=Counter)
    by_format: Counter[str] = field(default_factory=Counter)
    by_strategy: Counter[str] = field(default_factory=Counter)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def record(self, result: IngestResult) -> None:
        self.documents += 1
        self.chunks_written += result.chunks_written
        self.chunks_removed += result.chunks_removed
        self.by_state[str(result.state)] += 1
        self.by_format[result.fmt] += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "files_seen": self.files_seen,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "documents": self.documents,
            "chunks_written": self.chunks_written,
            "chunks_removed": self.chunks_removed,
            "by_state": dict(sorted(self.by_state.items())),
            "by_format": dict(sorted(self.by_format.items())),
            "by_strategy": dict(sorted(self.by_strategy.items())),
            "failures": self.failures[:20],
        }


def ingest_raw_document(
    doc: RawDocument,
    store: ChunkStore,
    index: ContentIndex,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    summary: IngestSummary | None = None,
) -> IngestResult:
    """Classify one already-loaded document, then chunk and store it if needed."""
    decision = index.classify(doc)

    if not decision.should_store:
        # UNCHANGED costs nothing to skip; DUPLICATE records who owns the text.
        return IngestResult(
            doc_id=doc.doc_id,
            state=decision.state,
            fmt=doc.fmt,
            duplicate_of=decision.duplicate_of,
        )

    removed = 0
    if decision.should_remove_old:
        # The document changed. Its old chunks describe text that no longer
        # exists, so they must go before the new ones are written.
        removed = store.delete_document(doc.doc_id)

    written = 0
    for strategy in strategies:
        chunks = chunk_document(doc, strategy=strategy)
        written += store.upsert(chunks)
        if summary is not None:
            summary.by_strategy[strategy] += len(chunks)

    index.record(doc)
    return IngestResult(
        doc_id=doc.doc_id,
        state=decision.state,
        fmt=doc.fmt,
        chunks_written=written,
        chunks_removed=removed,
    )


def ingest_file(
    path: Path,
    store: ChunkStore,
    index: ContentIndex,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    summary: IngestSummary | None = None,
) -> list[IngestResult]:
    """Ingest one file, which may contain many documents.

    A CSV of 298 deploys produces 298 results; every other format produces one.
    """
    documents = [enrich(doc) for doc in load_any(path)]
    return [ingest_raw_document(doc, store, index, strategies, summary) for doc in documents]


def delete_document(doc_id: str, store: ChunkStore, index: ContentIndex) -> int:
    """Remove a document and every chunk it produced."""
    removed = store.delete_document(doc_id)
    index.forget(doc_id)
    log.info("deleted %s (%d chunks)", doc_id, removed)
    return removed


def iter_corpus_files(root: Path, skip: Iterable[str] = ()) -> Iterator[Path]:
    """Every loadable file under a directory, in a stable order.

    Sorted so that two runs process files identically, which keeps duplicate
    detection deterministic: whichever document is seen first owns the text,
    and 'first' must not depend on filesystem ordering.
    """
    skip_names = set(skip)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in skip_names:
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def ingest_directory(
    root: Path,
    store: ChunkStore,
    index: ContentIndex,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    skip: Iterable[str] = (),
    summary: IngestSummary | None = None,
    log_every: int = 2000,
) -> IngestSummary:
    """Ingest every supported file under a directory.

    One unreadable file is logged and skipped. Aborting a pass over 67,000
    records because of a single corrupt PDF would be the wrong trade.
    """
    summary = summary or IngestSummary()

    for path in iter_corpus_files(root, skip=skip):
        summary.files_seen += 1
        try:
            results = ingest_file(path, store, index, strategies, summary)
        except UnsupportedFormatError:
            summary.files_skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            summary.files_failed += 1
            summary.failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            log.warning("failed to ingest %s: %s", path, exc)
            continue

        for result in results:
            summary.record(result)

        if summary.documents and summary.documents % log_every == 0:
            log.info(
                "%s documents -> %s chunks",
                f"{summary.documents:,}",
                f"{summary.chunks_written:,}",
            )

    return summary
