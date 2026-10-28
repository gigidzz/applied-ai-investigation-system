"""Tests for the two chunking strategies.

The load-bearing test is test_no_content_word_is_lost: it runs both strategies
over every document in the committed operations corpus and asserts that no
content word disappears. Two separate bugs were caught by exactly that check
during development — headings being discarded instead of carried into the
chunk text, once as a whole block and once when several sections packed into
one chunk.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from ragsys.ingestion.chunking import (
    UnknownStrategyError,
    chunk_document,
    fixed_windows,
    split_oversized_block,
)
from ragsys.ingestion.models import RawDocument

MARKDOWN_MARKER = re.compile(r"^#{1,6}$")


def document(text: str, fmt: str = "txt") -> RawDocument:
    return RawDocument(doc_id="test/doc", text=text, source_path="test/doc", fmt=fmt)


def test_short_text_becomes_one_chunk() -> None:
    doc = document("one two three four five")
    for strategy in ("structural", "fixed"):
        chunks = chunk_document(doc, strategy=strategy, size=160, overlap=40)
        assert len(chunks) == 1
        assert chunks[0].text == "one two three four five"


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_document(document("   \n  "), strategy="fixed") == []


def test_fixed_windows_have_the_requested_overlap() -> None:
    words = [f"w{i}" for i in range(250)]
    windows = fixed_windows(" ".join(words), size=100, overlap=20)

    # step = size - overlap = 80, so windows start at 0, 80 and 160. The third
    # already reaches word 249, so there is no fourth.
    assert len(windows) == 3
    assert windows[0].split()[0] == "w0"
    assert windows[1].split()[0] == "w80"
    assert windows[2].split()[0] == "w160"
    assert windows[2].split()[-1] == "w249"
    # The last 20 words of each window reappear at the start of the next.
    assert windows[0].split()[-20:] == windows[1].split()[:20]
    assert windows[1].split()[-20:] == windows[2].split()[:20]


def test_fixed_windows_cover_every_word() -> None:
    words = [f"w{i}" for i in range(97)]
    seen = {w for window in fixed_windows(" ".join(words), 20, 5) for w in window.split()}
    assert seen == set(words)


def test_structural_does_not_cut_mid_sentence() -> None:
    sentences = [f"This is sentence number {i} and it has a few words in it." for i in range(40)]
    doc = document(" ".join(sentences))

    for chunk in chunk_document(doc, strategy="structural", size=60, overlap=0):
        assert chunk.text.rstrip().endswith("."), chunk.text[-60:]


def test_structural_keeps_markdown_headings_in_the_text() -> None:
    text = (
        "# Runbook title\n\n"
        "## Connection pool sizing\n\n"
        "The minimum supported pool size is 25 connections per pod.\n\n"
        "## Escalation\n\n"
        "Page the on-call rotation first.\n"
    )
    joined = " ".join(c.text for c in chunk_document(document(text, "md"), "structural", 160, 40))

    for heading in ("Runbook title", "Connection pool sizing", "Escalation"):
        assert heading in joined


def test_structural_records_the_section_on_the_chunk() -> None:
    text = "# Title\n\n## Pool sizing\n\n" + " ".join(["word"] * 200) + "\n"
    chunks = chunk_document(document(text, "md"), "structural", size=80, overlap=20)
    assert any(c.metadata.get("section") == "Pool sizing" for c in chunks)


def test_a_single_oversized_sentence_falls_through_to_word_windows() -> None:
    # No sentence boundary to split on, so word windows are the only option.
    block = " ".join(["word"] * 300)
    pieces = split_oversized_block(block, size=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 100 for p in pieces)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(UnknownStrategyError, match="semantic"):
        chunk_document(document("some text"), strategy="semantic")


def test_overlap_at_least_size_is_rejected() -> None:
    # Step would be zero or negative, so the chunker would never advance.
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document("some text"), strategy="fixed", size=50, overlap=50)


def test_chunk_ids_are_stable_across_runs() -> None:
    doc = document(" ".join(["word"] * 400))
    first = [c.chunk_id for c in chunk_document(doc, "fixed", 100, 20)]
    second = [c.chunk_id for c in chunk_document(doc, "fixed", 100, 20)]
    assert first == second
    assert len(set(first)) == len(first)


def test_strategies_produce_different_chunk_ids_for_the_same_document() -> None:
    doc = document(" ".join(["word"] * 400))
    structural = {c.chunk_id for c in chunk_document(doc, "structural", 100, 20)}
    fixed = {c.chunk_id for c in chunk_document(doc, "fixed", 100, 20)}
    assert not structural & fixed


@pytest.mark.parametrize("strategy", ["structural", "fixed"])
def test_no_content_word_is_lost(strategy: str, ops_documents: list[RawDocument]) -> None:
    """Both strategies must preserve every content word of every document.

    Markdown '#' markers are excluded: they are markup, not content, and the
    structural strategy strips them while keeping the heading text.
    """
    assert ops_documents, "the committed operations corpus should not be empty"

    for doc in ops_documents:
        source = Counter(w for w in doc.text.split() if not MARKDOWN_MARKER.match(w))
        produced: Counter[str] = Counter()
        for chunk in chunk_document(doc, strategy=strategy):
            produced.update(chunk.text.split())

        missing = source - produced
        assert not missing, f"{doc.doc_id} [{strategy}] dropped {dict(list(missing.items())[:5])}"
