"""JSON Lines: the converted HotpotQA corpus, one record per line.

Each line is already an independent record, so each becomes one document.
The loader looks for a 'text' field and carries any id and title through,
rather than assuming our own field names, so it also handles a JSONL file
somebody else produced.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragsys.ingestion.loaders._paths import citation_path, document_id
from ragsys.ingestion.models import RawDocument

_TEXT_FIELDS = ("text", "content", "body", "passage")
_ID_FIELDS = ("doc_id", "id", "_id")


def load_jsonl(path: Path) -> list[RawDocument]:
    documents: list[RawDocument] = []

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            text = next((str(record[f]) for f in _TEXT_FIELDS if record.get(f)), "")
            if not text.strip():
                continue

            given_id = next((str(record[f]) for f in _ID_FIELDS if record.get(f)), None)
            doc_id = given_id or document_id(path, f"line={number}")

            metadata = {
                key: value
                for key, value in record.items()
                if key not in _TEXT_FIELDS and isinstance(value, str | int | float | bool)
            }
            metadata.setdefault("title", record.get("title") or doc_id)

            documents.append(
                RawDocument(
                    doc_id=doc_id,
                    text=text.strip(),
                    source_path=citation_path(path),
                    fmt="jsonl",
                    metadata=metadata,
                )
            )
    return documents
