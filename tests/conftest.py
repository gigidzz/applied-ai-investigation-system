"""Fixtures shared by the ingestion tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragsys.config import settings
from ragsys.ingestion.loaders import load_any
from ragsys.ingestion.metadata import enrich
from ragsys.ingestion.models import RawDocument
from ragsys.ingestion.pipeline import iter_corpus_files

OPS_SKIP = ("DESIGN.md", "MANIFEST.json")


@pytest.fixture(scope="session")
def ops_corpus_dir() -> Path:
    """The committed synthetic corpus.

    It is versioned precisely so the tests have real documents in six formats
    to run against, rather than hand-made fixtures that drift from reality.
    """
    path = settings.corpus_dir / "ops"
    if not path.is_dir():
        pytest.skip("operations corpus not present; run scripts/build_ops_corpus.py")
    return path


@pytest.fixture(scope="session")
def ops_documents(ops_corpus_dir: Path) -> list[RawDocument]:
    """Every operations document, loaded and enriched, excluding the change log.

    The CSV is excluded here because it alone contributes 298 documents, which
    would dominate any per-document test without adding coverage.
    """
    documents: list[RawDocument] = []
    for path in iter_corpus_files(ops_corpus_dir, skip=OPS_SKIP):
        if path.suffix == ".csv":
            continue
        documents.extend(enrich(doc) for doc in load_any(path))
    return documents
