"""CSV: the deploy change log.

The only format where one file is many documents. Each row is independently
meaningful and independently citable: answering "what was deployed on 14
March?" should cite one row, not a 298-row spreadsheet.

Each row becomes a readable sentence of 'column: value' pairs, because an
embedding model needs prose rather than comma-separated fields. The raw
column values are also kept as metadata, so they remain available for
filtering.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ragsys.ingestion.loaders._paths import citation_path, document_id
from ragsys.ingestion.models import RawDocument


def load_csv(path: Path) -> list[RawDocument]:
    documents: list[RawDocument] = []

    with path.open(encoding="utf-8", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=1):
            fields = {k: v for k, v in row.items() if k and v not in (None, "")}
            if not fields:
                continue
            text = "; ".join(f"{key}: {value}" for key, value in fields.items())
            documents.append(
                RawDocument(
                    doc_id=document_id(path, f"row={number}"),
                    text=text,
                    source_path=citation_path(path),
                    fmt="csv",
                    metadata={"title": f"{path.stem} row {number}", "row": number, **fields},
                )
            )
    return documents
