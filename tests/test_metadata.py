"""Tests for metadata extraction — the fields citations are built from."""

from __future__ import annotations

from ragsys.ingestion.metadata import extract
from ragsys.ingestion.models import RawDocument


def test_extracts_the_fields_an_incident_report_carries() -> None:
    text = (
        "INC-2026-03-14: Payment authorisation failures\n"
        "Northwind Payments - SEV1 - service payment-gateway - author s.petrova\n"
        "Release pay-gw-4.2.1 reduced the connection pool from 50 to 10."
    )
    found = extract(text)
    assert found["date"] == "2026-03-14"
    assert found["author"] == "s.petrova"
    assert found["severity"] == "SEV1"
    assert found["service"] == "payment-gateway"
    assert found["incident_refs"] == ["INC-2026-03-14"]
    assert found["releases"] == ["pay-gw-4.2.1"]


def test_a_date_deep_in_the_body_is_not_the_document_date() -> None:
    """A referenced date must not become the document's own date.

    A runbook that says "added after INC-2026-03-14" has no date of its own.
    Treating the reference as its date would wrongly boost that runbook for
    every query about March.
    """
    text = (
        "payment-gateway runbook\n\n"
        "## Connection pool sizing\n\n"
        + ("filler " * 120)
        + "A pre-deploy check now rejects pool sizes below the minimum, "
        "added after INC-2026-03-14."
    )
    found = extract(text)
    assert "date" not in found
    assert found["dates_mentioned"] == ["2026-03-14"]


def test_timeline_dates_are_separated_from_the_document_date() -> None:
    text = "Report\ndate: 2026-02-11\nSupersedes the 2026-01-05 decision."
    found = extract(text)
    assert found["date"] == "2026-02-11"
    assert found["dates_mentioned"] == ["2026-01-05"]


def test_iso_timestamps_are_matched_despite_the_time_suffix() -> None:
    # A trailing \b would fail here: there is no boundary between "4" and "T".
    assert extract("Started 2026-03-14T14:32:00Z")["date"] == "2026-03-14"


def test_the_word_after_service_is_not_mistaken_for_a_service_name() -> None:
    # Service names are hyphenated, which keeps ordinary prose out.
    assert "service" not in extract("the service that authorises card payments")
    assert extract("service payment-gateway")["service"] == "payment-gateway"


def test_absent_fields_are_omitted_rather_than_set_to_none() -> None:
    found = extract("A Wikipedia paragraph about a ghost town in Nevada.")
    assert found == {}


def test_loader_metadata_wins_over_extraction() -> None:
    """A CSV column or JSON key is authoritative; a regex is a best effort."""
    from ragsys.ingestion.metadata import enrich

    doc = RawDocument(
        doc_id="x",
        text="service payment-gateway - author s.petrova",
        source_path="x",
        fmt="csv",
        metadata={"service": "ledger-service"},
    )
    enriched = enrich(doc)
    assert enriched.metadata["service"] == "ledger-service"
    assert enriched.metadata["author"] == "s.petrova"


def test_references_are_deduplicated_in_first_seen_order() -> None:
    text = "See INC-2026-03-14, then INC-2026-03-02, then INC-2026-03-14 again."
    assert extract(text)["incident_refs"] == ["INC-2026-03-14", "INC-2026-03-02"]
