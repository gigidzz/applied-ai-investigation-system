"""PDF: incident post-mortems.

One document per page, so a citation can name the page a claim came from.
A PDF stores positioned glyphs rather than text, so extraction is inference
and a page can legitimately come back empty — a scan, or a page of images.
Those pages are skipped rather than stored as empty records.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ragsys.ingestion.loaders._paths import citation_path, document_id
from ragsys.ingestion.models import RawDocument


def load_pdf(path: Path) -> list[RawDocument]:
    reader = PdfReader(str(path))
    info = reader.metadata or {}
    documents: list[RawDocument] = []

    for number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - one bad page must not lose the rest
            text = ""
        if not text:
            continue
        documents.append(
            RawDocument(
                doc_id=document_id(path, f"page={number}"),
                text=text,
                source_path=citation_path(path),
                fmt="pdf",
                metadata={
                    "title": (info.get("/Title") or path.stem),
                    "page": number,
                    "page_count": len(reader.pages),
                },
            )
        )
    return documents
