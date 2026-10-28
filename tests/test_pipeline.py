"""End-to-end tests for ingestion: add, update, delete, and error handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragsys.ingestion.dedupe import ContentIndex, DocumentState
from ragsys.ingestion.models import RawDocument
from ragsys.ingestion.pipeline import (
    delete_document,
    ingest_directory,
    ingest_file,
    ingest_raw_document,
    iter_corpus_files,
)
from ragsys.ingestion.stores import InMemoryChunkStore, JsonlChunkStore


@pytest.fixture
def store() -> InMemoryChunkStore:
    return InMemoryChunkStore()


@pytest.fixture
def index() -> ContentIndex:
    return ContentIndex()


def document(doc_id: str, text: str) -> RawDocument:
    return RawDocument(doc_id=doc_id, text=text, source_path=doc_id, fmt="txt")


def test_add_then_add_again_does_no_work_the_second_time(
    store: InMemoryChunkStore, index: ContentIndex, ops_corpus_dir: Path
) -> None:
    pdf = ops_corpus_dir / "incidents" / "INC-2026-03-14.pdf"

    first = ingest_file(pdf, store, index)
    assert [r.state for r in first] == [DocumentState.NEW]
    assert first[0].chunks_written > 0
    after_first = store.count()

    second = ingest_file(pdf, store, index)
    assert [r.state for r in second] == [DocumentState.UNCHANGED]
    assert second[0].chunks_written == 0
    assert store.count() == after_first


def test_update_removes_the_old_chunks(store: InMemoryChunkStore, index: ContentIndex) -> None:
    original = document("doc", " ".join(["alpha"] * 400))
    ingest_raw_document(original, store, index)
    before = store.count()
    assert before > 0

    edited = document("doc", " ".join(["beta"] * 400))
    result = ingest_raw_document(edited, store, index)

    assert result.state is DocumentState.CHANGED
    assert result.chunks_removed == before
    # No chunk of the old text may survive, or the system would cite text
    # that no longer exists.
    assert all("alpha" not in c.text for c in store.chunks_for("doc"))


def test_delete_removes_every_chunk_and_forgets_the_document(
    store: InMemoryChunkStore, index: ContentIndex
) -> None:
    doc = document("doc", " ".join(["word"] * 300))
    ingest_raw_document(doc, store, index)
    assert store.count() > 0

    removed = delete_document("doc", store, index)
    assert removed > 0
    assert store.count() == 0
    assert len(index) == 0
    # Deleting then re-adding must work, not be treated as a duplicate.
    assert ingest_raw_document(doc, store, index).state is DocumentState.NEW


def test_one_file_can_produce_many_documents(
    store: InMemoryChunkStore, index: ContentIndex, ops_corpus_dir: Path
) -> None:
    results = ingest_file(ops_corpus_dir / "changes" / "change-log.csv", store, index)
    assert len(results) > 200, "each CSV row is an independent document"
    assert all(r.state is DocumentState.NEW for r in results)


def test_both_strategies_are_stored_side_by_side(
    store: InMemoryChunkStore, index: ContentIndex
) -> None:
    doc = document("doc", " ".join(["word"] * 500))
    ingest_raw_document(doc, store, index, strategies=("structural", "fixed"))

    strategies = {c.strategy for c in store.chunks_for("doc")}
    assert strategies == {"structural", "fixed"}


def test_one_unreadable_file_does_not_abort_the_run(
    store: InMemoryChunkStore, index: ContentIndex, tmp_path: Path
) -> None:
    (tmp_path / "good.txt").write_text("a readable document", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a PDF at all")
    (tmp_path / "also-good.md").write_text("# Title\n\nmore text", encoding="utf-8")

    summary = ingest_directory(tmp_path, store, index)

    assert summary.files_failed == 1
    assert summary.documents == 2, "the readable files must still be ingested"
    assert "broken.pdf" in summary.failures[0][0]


def test_unsupported_extensions_are_ignored_not_failed(
    store: InMemoryChunkStore, index: ContentIndex, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    (tmp_path / "sheet.xlsx").write_bytes(b"\x00\x01")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")

    summary = ingest_directory(tmp_path, store, index)
    assert summary.files_seen == 1
    assert summary.files_failed == 0


def test_corpus_files_are_yielded_in_a_stable_order(tmp_path: Path) -> None:
    """Duplicate detection depends on which document is seen first."""
    for name in ("c.txt", "a.txt", "b.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    once = [p.name for p in iter_corpus_files(tmp_path)]
    twice = [p.name for p in iter_corpus_files(tmp_path)]
    assert once == twice == sorted(once)


def test_jsonl_store_writes_one_record_per_line(index: ContentIndex, tmp_path: Path) -> None:
    out = tmp_path / "chunks.jsonl"
    doc = document("doc", " ".join(["word"] * 500))

    with JsonlChunkStore(out) as store:
        ingest_raw_document(doc, store, index)
        written = store.count()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == written
    record = json.loads(lines[0])
    assert {"chunk_id", "doc_id", "text", "index", "strategy"} <= record.keys()


def test_jsonl_store_refuses_deletion_with_a_clear_message(tmp_path: Path) -> None:
    with (
        JsonlChunkStore(tmp_path / "chunks.jsonl") as store,
        pytest.raises(NotImplementedError, match="append-only"),
    ):
        store.delete_document("doc")
