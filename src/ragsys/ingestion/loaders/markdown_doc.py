"""Markdown: runbooks and operations reports.

Markdown is the only format in the corpus with explicit structure — '#' marks
a title, '##' a section. The structural chunker uses those boundaries, so the
text is passed through unmodified rather than stripped of its markers.
"""

from __future__ import annotations

import re
from pathlib import Path

from ragsys.ingestion.loaders._paths import citation_path, document_id
from ragsys.ingestion.models import RawDocument

_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def load_markdown(path: Path) -> list[RawDocument]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    heading = _H1.search(text)
    return [
        RawDocument(
            doc_id=document_id(path),
            text=text,
            source_path=citation_path(path),
            fmt="md",
            metadata={"title": heading.group(1).strip() if heading else path.stem},
        )
    ]
