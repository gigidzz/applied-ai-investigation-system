"""Pull citable and filterable fields out of a document's text.

A loader knows what a file format is; it does not know what an incident report
looks like. This module reads the text and recovers the fields a citation
needs — date, author, severity, which service, which release — regardless of
which format the text arrived in.

The patterns are deliberately generic: a date is an ISO date, a release is a
dotted version with a prefix. Nothing here knows the names of the twelve
services in the operations corpus, so the same extraction works on a document
nobody has seen before. Where a format already supplies a field (a CSV column,
a JSON key, a PDF title), that value wins and extraction does not overwrite it.

Measured consequence worth remembering: ISO dates appear in 97.9% of the
operations records but only 0.6% of the whole corpus, because the Wikipedia
paragraphs carry a title and nothing else. Date filtering in retrieval must
therefore boost dated records, never exclude undated ones.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ragsys.ingestion.models import RawDocument

# A trailing \b would fail on "2026-03-14T14:20:00Z", because there is no word
# boundary between "4" and "T".
DATE = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")

# Usernames in this corpus are 'first-initial.surname'.
AUTHOR = re.compile(r"\bauthor[:\s]+([a-z]\.[a-z][a-z-]+)", re.IGNORECASE)

SEVERITY = re.compile(r"\bSEV([1-3])\b")

# Service names are multi-word-hyphenated, which avoids matching the English
# word following "service" in a sentence like "service that authorises...".
SERVICE = re.compile(r"\bservice[:\s]+([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b")

# Document references that let one document point at another.
INCIDENT_REF = re.compile(r"\b(INC-20\d{2}-\d{2}-\d{2})\b")
ADR_REF = re.compile(r"\b(ADR-\d{4})\b")

# A release is a hyphenated prefix plus a dotted version: 'pay-gw-4.2.1'.
RELEASE = re.compile(r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d+\.\d+\.\d+)\b")

MAX_REFS = 12

# A document's own date appears in its header: a title block, a metadata line,
# a 'date:' field. A date further into the body is almost always a reference to
# something else — "added after INC-2026-03-14", a timeline entry, a mention of
# another incident. Treating those as the document's date would wrongly boost a
# runbook for every March query, so only the header is trusted.
HEADER_CHARS = 400


def _unique(values: list[str]) -> list[str]:
    """Deduplicate while preserving first-seen order, so output is stable."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def extract(text: str) -> dict[str, object]:
    """Fields recovered from the text alone. Absent fields are omitted."""
    found: dict[str, object] = {}

    header_dates = _unique(DATE.findall(text[:HEADER_CHARS]))
    if header_dates:
        found["date"] = header_dates[0]

    all_dates = _unique(DATE.findall(text))
    mentioned = [d for d in all_dates if d != found.get("date")]
    if mentioned:
        found["dates_mentioned"] = mentioned[:MAX_REFS]

    if match := AUTHOR.search(text):
        found["author"] = match.group(1).lower()

    if match := SEVERITY.search(text):
        found["severity"] = f"SEV{match.group(1)}"

    if match := SERVICE.search(text):
        found["service"] = match.group(1)

    if incidents := _unique(INCIDENT_REF.findall(text)):
        found["incident_refs"] = incidents[:MAX_REFS]

    if adrs := _unique(ADR_REF.findall(text)):
        found["adr_refs"] = adrs[:MAX_REFS]

    if releases := _unique(RELEASE.findall(text)):
        found["releases"] = releases[:MAX_REFS]

    return found


def enrich(doc: RawDocument) -> RawDocument:
    """Return the document with extracted fields merged into its metadata.

    Values the loader already supplied are kept: a CSV column or a JSON key is
    authoritative, while a regex over prose is a best effort.
    """
    extracted = extract(doc.text)
    merged = {**extracted, **{k: v for k, v in doc.metadata.items() if v is not None}}
    return replace(doc, metadata=merged)
