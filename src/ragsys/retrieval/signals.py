"""Ranking signals that similarity cannot supply.

Fusion repairs disagreement between the two retrieval routes. It cannot repair
agreement on the wrong answer, and a measured query showed exactly that: asked
"when did the rollback finish", both routes returned generic rollback
*runbooks*, and neither returned the incident report or the chat thread that
actually record a completion time. Neither engine malfunctioned — the question
genuinely is more similar to a document about rollback procedure than to a
timeline entry that happens to mention one finishing.

What is missing is not a better similarity measure but knowledge that "when
did X finish" asks for a record rather than a procedure. These signals supply
that, along with three others, each added in response to a failure observed on
this corpus rather than anticipated in advance.

Scores are applied multiplicatively:

    final = fusion x (1 + w1*metadata + w2*type + w3*recency - w4*crowding)

Multiplicative because fusion scores are small and their scale depends on the
rank-fusion constant, so an additive bonus would have to be retuned whenever
that changed. As a multiplier, a weight of 0.6 means "a full metadata match is
worth a 60% boost" regardless of the underlying scale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from ragsys.ingestion.metadata import ADR_REF, DATE, INCIDENT_REF, RELEASE, SEVERITY

# Document kinds, taken from the corpus layout rather than guessed: a path of
# the form ops/<kind>/... names its own kind, and anything else is encyclopedic
# prose with no operational role.
WIKI = "wiki"
OPS_PREFIX = "ops/"


class QuestionKind(StrEnum):
    """What shape of document a question is really asking for."""

    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"
    DEPENDENCY = "dependency"
    FACTUAL = "factual"


_TEMPORAL = re.compile(
    r"\b(when|what time|how long|at what point|timeline|"
    r"(?:did|was).{0,24}\b(?:finish|complete|start|begin|resolve|end))\b",
    re.IGNORECASE,
)
_PROCEDURAL = re.compile(
    r"\b(how (?:do|to|should)|what steps|procedure|runbook|playbook|"
    r"what should i do|mitigat|remediat)\w*\b",
    re.IGNORECASE,
)
# A service name is a hyphenated lowercase compound: payment-gateway,
# ledger-service. Releases and incident references have the same shape once
# their trailing version or date is ignored, so they are removed before this
# pattern is applied or every release would register as a service mention.
_SERVICE_LIKE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
_IDENTIFIER_SHAPED = re.compile(
    r"\b(?:inc-20\d{2}-\d{2}-\d{2}|adr-\d{4}"
    r"|[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d+\.\d+\.\d+)\b",
    re.IGNORECASE,
)

_DEPENDENCY = re.compile(
    r"\b(depend\w*|upstream|downstream|owns?|owner|which team|tier|"
    r"affected by|impact\w*)\b",
    re.IGNORECASE,
)

# A "when" question wants a dated record, so procedure documents are pushed
# down rather than merely not boosted: a runbook is a plausible match for the
# words and a poor match for the intent, which is the failure mode observed.
TYPE_PREFERENCE: dict[QuestionKind, dict[str, float]] = {
    QuestionKind.TEMPORAL: {
        "incidents": 1.0,
        "threads": 0.9,
        "changes": 0.8,
        "reports": 0.3,
        "adr": 0.0,
        "services": -0.3,
        "runbooks": -0.8,
        WIKI: 0.0,
    },
    QuestionKind.PROCEDURAL: {
        "runbooks": 1.0,
        "adr": 0.3,
        "incidents": 0.1,
        "threads": -0.2,
        "changes": -0.4,
        "services": 0.0,
        "reports": 0.0,
        WIKI: 0.0,
    },
    QuestionKind.DEPENDENCY: {
        "services": 1.0,
        "adr": 0.4,
        "incidents": 0.2,
        "runbooks": 0.1,
        "reports": 0.1,
        "threads": -0.1,
        "changes": -0.2,
        WIKI: 0.0,
    },
    QuestionKind.FACTUAL: {},
}

# How far back a date still earns some boost. A year, because the operations
# corpus spans 2026 and a tighter window would make most of it invisible.
RECENCY_WINDOW_DAYS = 365


@dataclass(frozen=True)
class RerankWeights:
    """How much each signal is allowed to move a result.

    Held in one object so the evaluation harness can compare weight sets as
    configurations without touching the retriever.
    """

    metadata: float = 0.60
    doc_type: float = 0.35
    recency: float = 0.15
    crowding: float = 0.25

    @classmethod
    def off(cls) -> RerankWeights:
        """All signals disabled: fusion only, for the configuration comparison."""
        return cls(metadata=0.0, doc_type=0.0, recency=0.0, crowding=0.0)


def classify_question(question: str) -> QuestionKind:
    """First match wins, most specific pattern first.

    Order matters: "how long did the rollback take" is temporal despite
    starting with "how", so the temporal test runs before the procedural one.
    """
    if _TEMPORAL.search(question):
        return QuestionKind.TEMPORAL
    if _PROCEDURAL.search(question):
        return QuestionKind.PROCEDURAL
    if _DEPENDENCY.search(question):
        return QuestionKind.DEPENDENCY
    return QuestionKind.FACTUAL


def document_kind(payload: dict[str, Any]) -> str:
    path = str(payload.get("meta_source_path", ""))
    if path.startswith(OPS_PREFIX):
        parts = path.split("/")
        if len(parts) > 1:
            return parts[1]
    return WIKI


@dataclass(frozen=True)
class QueryEntities:
    """Identifiers named explicitly in a question."""

    releases: frozenset[str]
    incidents: frozenset[str]
    adrs: frozenset[str]
    dates: frozenset[str]
    severities: frozenset[str]

    def __bool__(self) -> bool:
        return bool(self.releases or self.incidents or self.adrs or self.dates or self.severities)


def query_entities(question: str) -> QueryEntities:
    """Reuses the ingestion patterns, so a question is read the same way a document is.

    Incident references are removed before dates are read. `INC-2026-03-14`
    contains `2026-03-14`, and counting that as a date the question asks about
    adds an expectation nothing can satisfy: a chunk tagged with the incident
    but carrying no date field would score a partial match for what is in fact
    an exact one.
    """
    upper = question.upper()
    lowered = question.lower()
    without_identifiers = ADR_REF.sub(" ", INCIDENT_REF.sub(" ", upper))
    return QueryEntities(
        releases=frozenset(RELEASE.findall(lowered)),
        incidents=frozenset(INCIDENT_REF.findall(upper)),
        adrs=frozenset(ADR_REF.findall(upper)),
        dates=frozenset(DATE.findall(without_identifiers)),
        severities=frozenset(SEVERITY.findall(upper)),
    )


def question_names_a_service(question: str) -> bool:
    """Whether the question refers to any service at all.

    Needed because the service check is otherwise asymmetric with the others:
    a release or an incident is only expected of a chunk when the question
    names one, but a chunk declares its service regardless. Without this, a
    chunk matching a release exactly scored 0.56 rather than 1.0 purely for
    carrying a service tag the question never mentioned.
    """
    stripped = _IDENTIFIER_SHAPED.sub(" ", question.lower())
    return bool(_SERVICE_LIKE.search(stripped))


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return {str(value)}


def metadata_agreement(question: str, payload: dict[str, Any]) -> float:
    """How strongly a chunk's extracted metadata matches the question.

    Returns 0 to 1. This is the signal neither retrieval route can produce:
    both see only chunk text, and neither knows a chunk is *tagged*
    `service: payment-gateway`.
    """
    entities = query_entities(question)
    lowered = question.lower()
    hits = 0.0
    possible = 0.0

    def check(weight: float, matched: bool) -> None:
        nonlocal hits, possible
        possible += weight
        if matched:
            hits += weight

    # An exact identifier is the strongest evidence a question can carry.
    if entities.releases:
        declared = {r.lower() for r in _as_set(payload.get("meta_releases"))}
        check(1.0, bool(entities.releases & declared))
    if entities.incidents:
        declared = {r.upper() for r in _as_set(payload.get("meta_incident_refs"))}
        check(1.0, bool(entities.incidents & declared))
    if entities.adrs:
        declared = {r.upper() for r in _as_set(payload.get("meta_adr_refs"))}
        check(0.8, bool(entities.adrs & declared))
    if entities.dates:
        declared = _as_set(payload.get("meta_date")) | _as_set(payload.get("meta_dates_mentioned"))
        check(0.7, bool(entities.dates & declared))
    if entities.severities:
        declared = {s.upper() for s in _as_set(payload.get("meta_severity"))}
        check(0.5, any(f"SEV{s}" in declared for s in entities.severities))

    # Service names are tested the other way round: rather than guessing which
    # words in a question are a service, take the name the chunk declares and
    # look for it. Both spellings are checked because a question says "payment
    # gateway" where the corpus writes "payment-gateway".
    #
    # The check only counts when the chunk's service is actually named, or
    # when the question names some other service and this chunk is therefore
    # about the wrong one. A question mentioning no service at all is not
    # asking about services, so an unmatched service field is not a miss.
    service = payload.get("meta_service")
    if service:
        name = str(service).lower()
        matched = name in lowered or name.replace("-", " ") in lowered
        if matched or question_names_a_service(question):
            check(0.8, matched)

    if possible == 0.0:
        return 0.0
    return hits / possible


def type_preference(kind: QuestionKind, payload: dict[str, Any]) -> float:
    """Between -1 and 1. Negative actively demotes the wrong shape of document."""
    return TYPE_PREFERENCE.get(kind, {}).get(document_kind(payload), 0.0)


def recency(payload: dict[str, Any], reference: date | None = None) -> float:
    """Between 0 and 1, and never negative.

    Absence of a date scores 0, not a penalty. Dates appear in 97.9% of
    operations records but only 0.6% of the corpus, so penalising their absence
    would suppress all 66,581 Wikipedia paragraphs on any query the signal
    happened to apply to.
    """
    raw = payload.get("meta_date")
    if not raw:
        return 0.0
    try:
        parsed = date.fromisoformat(str(raw)[:10])
    except ValueError:
        return 0.0
    anchor = reference or date.today()
    age = (anchor - parsed).days
    if age < 0:
        return 1.0
    if age >= RECENCY_WINDOW_DAYS:
        return 0.0
    return 1.0 - age / RECENCY_WINDOW_DAYS
