"""Tests for the four document states."""

from __future__ import annotations

from ragsys.ingestion.dedupe import ContentIndex, DocumentState
from ragsys.ingestion.models import RawDocument


def document(doc_id: str, text: str) -> RawDocument:
    return RawDocument(doc_id=doc_id, text=text, source_path=doc_id, fmt="txt")


def test_an_unseen_document_is_new() -> None:
    index = ContentIndex()
    decision = index.classify(document("a", "hello"))
    assert decision.state is DocumentState.NEW
    assert decision.should_store
    assert not decision.should_remove_old


def test_the_same_document_twice_is_unchanged() -> None:
    index = ContentIndex()
    doc = document("a", "hello")
    index.record(doc)

    decision = index.classify(doc)
    assert decision.state is DocumentState.UNCHANGED
    assert not decision.should_store, "re-embedding identical text is the waste to avoid"


def test_edited_text_under_the_same_id_is_changed() -> None:
    index = ContentIndex()
    index.record(document("a", "hello"))

    decision = index.classify(document("a", "hello, world"))
    assert decision.state is DocumentState.CHANGED
    assert decision.should_store
    assert decision.should_remove_old, "stale chunks must go before new ones are written"


def test_identical_text_under_a_different_id_is_a_duplicate() -> None:
    """The case identity-by-name cannot catch.

    Stage 1 deduplicated HotpotQA by title, because its labels are
    title-based. Profiling then found 18 distinct texts published under more
    than one title. Hashing the content catches them.
    """
    index = ContentIndex()
    index.record(document("original", "the same words exactly"))

    decision = index.classify(document("copy", "the same words exactly"))
    assert decision.state is DocumentState.DUPLICATE
    assert decision.duplicate_of == "original"
    assert not decision.should_store


def test_a_document_may_reclaim_its_own_text_after_being_edited_back() -> None:
    index = ContentIndex()
    index.record(document("a", "first"))
    index.record(document("a", "second"))

    # 'first' is no longer owned by anything, so another document may take it.
    assert index.classify(document("b", "first")).state is DocumentState.NEW


def test_forget_frees_both_the_id_and_the_content() -> None:
    index = ContentIndex()
    index.record(document("a", "some text"))
    assert len(index) == 1

    index.forget("a")
    assert len(index) == 0
    assert index.classify(document("a", "some text")).state is DocumentState.NEW
    assert index.classify(document("b", "some text")).state is DocumentState.NEW


def test_rebuilding_from_stored_records() -> None:
    """Stage 3 reconstructs this index from the vector store on startup."""
    doc = document("a", "some text")
    index = ContentIndex.from_records([(doc.doc_id, doc.hash)])

    assert index.classify(doc).state is DocumentState.UNCHANGED
    assert index.classify(document("b", "some text")).duplicate_of == "a"
