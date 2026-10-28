"""Tests for the format loaders, run against the committed operations corpus.

Real documents rather than hand-made fixtures, because the point of these
loaders is to survive real files: a PDF whose text is positioned glyphs, a
DOCX that is zipped XML, a CSV that is 298 records in one file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsys.ingestion.loaders import (
    SUPPORTED_SUFFIXES,
    UnsupportedFormatError,
    load_any,
    loader_for,
)
from ragsys.ingestion.loaders.json_doc import flatten


def test_every_expected_format_has_a_loader() -> None:
    for suffix in (".pdf", ".docx", ".csv", ".md", ".json", ".jsonl", ".txt"):
        assert suffix in SUPPORTED_SUFFIXES
        assert loader_for(Path(f"x{suffix}")) is not None


def test_unsupported_format_names_what_is_supported() -> None:
    with pytest.raises(UnsupportedFormatError, match="xlsx"):
        loader_for(Path("spreadsheet.xlsx"))


def test_pdf_text_is_extractable_and_carries_the_page(ops_corpus_dir: Path) -> None:
    docs = load_any(ops_corpus_dir / "incidents" / "INC-2026-03-14.pdf")
    assert len(docs) == 1  # one document per page; these reports are one page

    doc = docs[0]
    assert doc.fmt == "pdf"
    assert doc.metadata["page"] == 1
    assert "page=1" in doc.doc_id
    # The facts the flagship question depends on must survive extraction.
    assert "pay-gw-4.2.1" in doc.text
    assert "503" in doc.text


def test_docx_text_and_core_properties(ops_corpus_dir: Path) -> None:
    doc = load_any(ops_corpus_dir / "adr" / "ADR-0007.docx")[0]
    assert doc.fmt == "docx"
    assert doc.metadata["author"] == "t.lindqvist"
    assert "settlement-queue" in doc.text


def test_csv_yields_one_document_per_row(ops_corpus_dir: Path) -> None:
    docs = load_any(ops_corpus_dir / "changes" / "change-log.csv")
    assert len(docs) > 200, "the change log should contribute hundreds of records"

    # Each row is independently citable, so each gets its own id.
    assert len({d.doc_id for d in docs}) == len(docs)
    assert all("row=" in d.doc_id for d in docs)

    # Columns are kept as metadata as well as rendered into the text.
    flagship = next(d for d in docs if "pay-gw-4.2.1" in d.text)
    assert flagship.metadata["service"] == "payment-gateway"
    assert "release: pay-gw-4.2.1" in flagship.text


def test_markdown_title_comes_from_the_first_heading(ops_corpus_dir: Path) -> None:
    doc = load_any(ops_corpus_dir / "runbooks" / "runbook-payment-gateway-v2.md")[0]
    assert doc.metadata["title"] == "payment-gateway runbook (v2)"
    # Heading markers are preserved for the structural chunker to use.
    assert "## " in doc.text


def test_json_is_flattened_into_readable_lines(ops_corpus_dir: Path) -> None:
    doc = load_any(ops_corpus_dir / "services" / "payment-gateway.json")[0]
    assert "name: payment-gateway" in doc.text
    assert "team: Payments Core" in doc.text
    # A list becomes a comma-separated value rather than JSON syntax.
    assert "depended_on_by: checkout-api, merchant-portal" in doc.text
    assert "{" not in doc.text
    # Scalars are also available as metadata for filtering.
    assert doc.metadata["team"] == "Payments Core"


def test_flatten_handles_nesting_and_empty_values() -> None:
    assert flatten({"a": 1}) == ["a: 1"]
    assert flatten({"a": {"b": 2}}) == ["a.b: 2"]
    assert flatten({"a": [1, 2]}) == ["a: 1, 2"]
    assert flatten({"a": []}) == []
    assert flatten({"a": [{"b": 1}]}) == ["a[0].b: 1"]


def test_text_loader_keeps_the_whole_thread(ops_corpus_dir: Path) -> None:
    doc = load_any(ops_corpus_dir / "threads" / "THREAD-2026-03-14-incident.txt")[0]
    assert doc.fmt == "txt"
    # Both sides of planted contradiction C1 live in this file.
    assert "15:40" in doc.text
    assert "April event" in doc.text


def test_jsonl_yields_one_document_per_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"doc_id": "a", "title": "A", "text": "first record"}\n'
        '{"doc_id": "b", "title": "B", "text": "second record"}\n'
        "\n"  # blank lines are skipped
        '{"doc_id": "c", "title": "C", "text": "   "}\n',  # empty text is skipped
        encoding="utf-8",
    )
    docs = load_any(path)
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].metadata["title"] == "A"


def test_loaders_skip_empty_files(tmp_path: Path) -> None:
    for name in ("empty.txt", "empty.md"):
        (tmp_path / name).write_text("   \n", encoding="utf-8")
        assert load_any(tmp_path / name) == []
