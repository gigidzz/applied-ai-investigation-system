"""Retrieval and answer metrics, each chosen for a failure it catches.

Nothing here calls a model. These are arithmetic over lists, so they are fast,
deterministic, and testable against worked examples — which matters because a
metric that is itself wrong invalidates every number in the report.

The two corpora identify documents differently: HotpotQA labels name a derived
paragraph id, while the authored questions name a path suffix such as
`incidents/INC-2026-03-14.pdf`. Both are compared with the same matcher so one
harness can score both without special cases at the call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def matches(gold: str, candidate: str) -> bool:
    """Whether a retrieved identifier refers to the gold document.

    Exact for HotpotQA's `hotpot:...` ids, suffix for the authored questions,
    whose keys deliberately omit the corpus prefix so the answer key does not
    have to change if the corpus moves.
    """
    if not gold or not candidate:
        return False
    return candidate == gold or candidate.endswith(gold)


def found_at(gold: str, retrieved: Sequence[str]) -> int | None:
    """One-based rank of the first match, or None."""
    for position, candidate in enumerate(retrieved, start=1):
        if matches(gold, candidate):
            return position
    return None


def recall_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of the needed documents present in the top k.

    Catches the failure no prompt can repair: evidence that never arrived.
    """
    if not gold:
        return 0.0
    window = retrieved[:k]
    hits = sum(1 for g in gold if any(matches(g, c) for c in window))
    return hits / len(gold)


def mrr(gold: Sequence[str], retrieved: Sequence[str]) -> float:
    """Reciprocal rank of the first correct document.

    Recall treats rank 1 and rank 15 alike, but only the top of the list is
    read closely by the model, so a separate measure of *how high* is needed.
    """
    ranks = [r for r in (found_at(g, retrieved) for g in gold) if r is not None]
    return 1.0 / min(ranks) if ranks else 0.0


def precision_at_k(gold: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of the top k that are gold documents.

    Low precision is not necessarily bad here — a chunk may be useful context
    without being in the answer key — so this is reported, not optimised.
    """
    window = retrieved[:k]
    if not window:
        return 0.0
    hits = sum(1 for c in window if any(matches(g, c) for g in gold))
    return hits / len(window)


@dataclass(frozen=True)
class RetrievalScore:
    """One question's retrieval result."""

    question_id: str
    category: str
    recall_at_5: float
    recall_at_10: float
    recall_at_15: float
    mrr: float
    precision_at_15: float
    rank_of_first: int | None
    distinct_documents: int
    complete: bool

    @classmethod
    def compute(
        cls,
        question_id: str,
        category: str,
        gold: Sequence[str],
        retrieved: Sequence[str],
    ) -> RetrievalScore:
        ranks = [r for r in (found_at(g, retrieved) for g in gold) if r is not None]
        return cls(
            question_id=question_id,
            category=category,
            recall_at_5=recall_at_k(gold, retrieved, 5),
            recall_at_10=recall_at_k(gold, retrieved, 10),
            recall_at_15=recall_at_k(gold, retrieved, 15),
            mrr=mrr(gold, retrieved),
            precision_at_15=precision_at_k(gold, retrieved, 15),
            rank_of_first=min(ranks) if ranks else None,
            distinct_documents=len(set(retrieved)),
            complete=bool(gold) and len(ranks) == len(gold),
        )


def citation_validity(cited_labels: Sequence[str], invalid_labels: Sequence[str]) -> float:
    """Fraction of citations that point at a source that was actually supplied.

    This is the fabrication check, and it is arithmetic rather than judgement:
    a label the model invented either was or was not in the supplied set.
    """
    total = len(cited_labels) + len(invalid_labels)
    if total == 0:
        return 0.0
    return len(cited_labels) / total


def citation_relevance(cited_documents: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of cited documents that the answer key names.

    Deliberately reported rather than maximised. A correct answer may cite a
    document the key omits, which was observed: the synthesis question was
    answered correctly from the incident report and the monthly report while
    the key names the change log and the service catalogue.
    """
    if not cited_documents:
        return 0.0
    hits = sum(1 for c in cited_documents if any(matches(g, c) for g in gold))
    return hits / len(cited_documents)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
