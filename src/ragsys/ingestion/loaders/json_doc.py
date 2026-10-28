"""JSON: the service catalogue.

JSON is structure, not prose. Embedding the literal characters of
'{"tier": 1}' produces a vector about punctuation, so the object is first
flattened into readable lines:

    name: payment-gateway
    team: Payments Core
    depended_on_by: checkout-api, merchant-portal

Scalar top-level fields are also kept as metadata for filtering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragsys.ingestion.loaders._paths import citation_path, document_id
from ragsys.ingestion.models import RawDocument


def flatten(value: Any, prefix: str = "") -> list[str]:
    """Render nested JSON as one 'dotted.key: value' line per leaf."""
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            lines.extend(flatten(item, f"{prefix}{key}." if prefix else f"{key}."))
        return lines
    if isinstance(value, list):
        if not value:
            return []
        if all(not isinstance(v, dict | list) for v in value):
            return [f"{prefix.rstrip('.')}: {', '.join(str(v) for v in value)}"]
        lines = []
        for position, item in enumerate(value):
            lines.extend(flatten(item, f"{prefix.rstrip('.')}[{position}]."))
        return lines
    return [f"{prefix.rstrip('.')}: {value}"]


def load_json(path: Path) -> list[RawDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = flatten(data)
    if not lines:
        return []

    scalars = (
        {k: v for k, v in data.items() if isinstance(v, str | int | float | bool)}
        if isinstance(data, dict)
        else {}
    )
    return [
        RawDocument(
            doc_id=document_id(path),
            text="\n".join(lines),
            source_path=citation_path(path),
            fmt="json",
            metadata={"title": str(scalars.get("name") or path.stem), **scalars},
        )
    ]
