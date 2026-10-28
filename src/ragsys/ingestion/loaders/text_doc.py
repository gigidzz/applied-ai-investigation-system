"""Plain text: chat threads and ticket exports."""

from __future__ import annotations

from pathlib import Path

from ragsys.ingestion.loaders._paths import citation_path, document_id
from ragsys.ingestion.models import RawDocument


def load_text(path: Path) -> list[RawDocument]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [
        RawDocument(
            doc_id=document_id(path),
            text=text,
            source_path=citation_path(path),
            fmt="txt",
            metadata={"title": path.stem},
        )
    ]
