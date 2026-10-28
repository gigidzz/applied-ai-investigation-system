"""One loader per file format, behind a single interface.

Every loader has the signature:

    load(path: Path) -> list[RawDocument]

It returns a list rather than a single document because one file does not
always mean one record. A CSV of 298 deploys yields 298 documents, because
each row answers a question on its own. A multi-page PDF yields one per page,
so a citation can name the page a claim came from.

Loaders raise on a file they cannot read. The pipeline decides whether that
aborts the run or is logged and skipped; a single corrupt file should not end
a pass over 67,000 records.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ragsys.ingestion.loaders.csv_rows import load_csv
from ragsys.ingestion.loaders.docx_doc import load_docx
from ragsys.ingestion.loaders.json_doc import load_json
from ragsys.ingestion.loaders.jsonl_records import load_jsonl
from ragsys.ingestion.loaders.markdown_doc import load_markdown
from ragsys.ingestion.loaders.pdf_pages import load_pdf
from ragsys.ingestion.loaders.text_doc import load_text
from ragsys.ingestion.models import RawDocument

Loader = Callable[[Path], list[RawDocument]]

LOADERS: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".csv": load_csv,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".json": load_json,
    ".jsonl": load_jsonl,
    ".txt": load_text,
}

SUPPORTED_SUFFIXES = tuple(sorted(LOADERS))


class UnsupportedFormatError(ValueError):
    """Raised for a file extension no loader handles."""


def loader_for(path: Path) -> Loader:
    suffix = path.suffix.lower()
    try:
        return LOADERS[suffix]
    except KeyError:
        raise UnsupportedFormatError(
            f"no loader for '{suffix}' ({path.name}); supported: {', '.join(SUPPORTED_SUFFIXES)}"
        ) from None


def load_any(path: Path) -> list[RawDocument]:
    """Dispatch to the right loader for this file's extension."""
    return loader_for(path)(path)


__all__ = [
    "LOADERS",
    "SUPPORTED_SUFFIXES",
    "UnsupportedFormatError",
    "load_any",
    "loader_for",
]
