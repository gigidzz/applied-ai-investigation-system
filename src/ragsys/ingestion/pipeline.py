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

from ragsys.config import settings
from ragsys.ingestion.chunking import STRATEGIES, chunk_document
from ragsys.ingestion.dedupe import ContentIndex, DocumentState
from ragsys.ingestion.loaders import SUPPORTED_SUFFIXES, UnsupportedFormatError, load_any
from ragsys.ingestion.loaders._paths import citation_path
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


# Uploads live under the corpus root so their ids read like every other
# document's — "uploads/report.pdf" rather than an absolute temp path that
# would differ on every machine and be meaningless in a citation.
UPLOAD_DIRNAME = "uploads"


class UnsupportedUpload(ValueError):
    """The uploaded file has no loader."""


def upload_dir() -> Path:
    return settings.corpus_dir / UPLOAD_DIRNAME


def ingest_upload(
    filename: str,
    data: bytes,
    store: ChunkStore,
    index: ContentIndex,
    strategies: Sequence[str] | None = None,
) -> list[IngestResult]:
    """Ingest a file supplied as bytes, as an upload is.

    Lives here rather than in the web layer because it is an ingestion
    concern: a command-line importer would want exactly this. Keeping it in
    the domain also means it is tested without starting a server.

    Re-uploading the same filename is an update, not a duplicate: the
    document id is derived from the path, so change detection sees the same
    id with new content and replaces the previous version's chunks. That is
    the intended behaviour and the reason uploads are not given unique names.

    Chunked with one strategy, not both. The offline build writes every
    strategy so that stage 6 can compare them, but a live index is built with
    exactly one and its BM25 statistics are computed over that one. Writing
    both here would store each uploaded passage twice under two chunkings, so
    an uploaded document would occupy twice the result slots of a document
    that arrived in the bulk build, and retrieval would surface two near
    duplicates of the same text.
    """
    strategies = strategies or (settings.chunk_strategy,)
    # Only the final component is used. A browser can send a name containing
    # path separators, and joining that to a directory would write outside it.
    safe_name = Path(filename).name
    if not safe_name:
        raise UnsupportedUpload("the upload has no filename")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedUpload(
            f"{suffix or 'no extension'} is not supported; "
            f"expected one of {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    target = upload_dir() / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    # Replace rather than add. Change detection alone is not enough here: if
    # the new version of a PDF has fewer pages than the old one, the extra
    # pages' chunks belong to document ids the new file no longer produces,
    # so nothing would ever update or remove them and the system would keep
    # citing pages that no longer exist.
    #
    # The order matters and an earlier version got it wrong. Deleting the
    # chunks and then calling the normal pipeline lost the document outright
    # when the file was re-uploaded unchanged: change detection compared the
    # content against what it had already recorded, concluded UNCHANGED, and
    # skipped the write that would have restored what the delete removed. So
    # the previous state is discarded first, which makes every document in
    # the upload NEW and guarantees the write happens.
    documents = [enrich(doc) for doc in load_any(target)]
    for document in documents:
        index.forget(document.doc_id)

    citation = citation_path(target)
    removed = store.delete_source(citation)
    if removed:
        log.info("replacing %s: removed %d chunks of the previous version", citation, removed)

    log.info("ingesting upload %s (%d bytes, %d documents)", safe_name, len(data), len(documents))
    return [ingest_raw_document(doc, store, index, strategies) for doc in documents]


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
