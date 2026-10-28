"""The answer object, and the check that its citations are real.

A model asked to cite its sources will sometimes cite a source it was never
given. That is not a judgement call about quality — it is a verifiable fact
about a string, so it is checked in code rather than trusted or sent to a
judge. Every label the answer claims is matched against the labels actually
supplied, and anything left over is recorded as invalid.

Keeping the invalid labels rather than silently dropping them matters: an
answer that cites S18 when fifteen sources were given is evidence about the
system, and the evaluation harness needs to count it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Citations are written as [S3], and several may share one bracket: [S1][S4].
CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True)
class Citation:
    """A resolved reference from the answer back to a retrieved chunk."""

    label: str
    chunk_id: str
    doc_id: str
    source_path: str

    def __str__(self) -> str:
        return f"{self.label} -> {self.source_path}"


@dataclass(frozen=True)
class Contradiction:
    """Two or more sources the model reports as disagreeing."""

    description: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Answer:
    """Everything produced for one question, including what went wrong."""

    question: str
    text: str
    citations: tuple[Citation, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    abstained: bool = False
    abstain_reason: str = ""
    invalid_labels: tuple[str, ...] = ()
    confidence: str = "low"
    steps: tuple[str, ...] = ()
    context_size: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)

    @property
    def cited_documents(self) -> tuple[str, ...]:
        """Distinct source documents, in the order first cited."""
        seen: dict[str, None] = {}
        for citation in self.citations:
            seen.setdefault(citation.source_path, None)
        return tuple(seen)

    @property
    def is_grounded(self) -> bool:
        """Every claim traceable, or an explicit refusal to claim anything.

        An abstention is grounded by definition: it asserts nothing. An answer
        with no citations at all is not, even if it happens to be correct,
        because nothing in it can be checked.
        """
        if self.failed:
            return False
        if self.invalid_labels:
            return False
        return self.abstained or bool(self.citations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "citations": [
                {
                    "label": c.label,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source_path": c.source_path,
                }
                for c in self.citations
            ],
            "contradictions": [
                {"description": c.description, "labels": list(c.labels)}
                for c in self.contradictions
            ],
            "invalid_labels": list(self.invalid_labels),
            "cited_documents": list(self.cited_documents),
            "confidence": self.confidence,
            "steps": list(self.steps),
            "context_size": self.context_size,
            "tool_calls": self.tool_calls,
            "is_grounded": self.is_grounded,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


def labels_in(text: str) -> list[str]:
    """Citation labels appearing in the answer, in order, without repeats."""
    return list(dict.fromkeys(CITATION_PATTERN.findall(text)))


def resolve_citations(
    text: str,
    supplied: list[dict[str, Any]],
    claimed: list[str] | None = None,
) -> tuple[tuple[Citation, ...], tuple[str, ...]]:
    """Match labels against the sources that were actually given.

    Labels are taken from the answer text, and from the model's own list if it
    provided one. The text is authoritative about what is being claimed, but a
    label listed and not used still asserts a source was relevant, so both are
    checked and reported together.
    """
    by_label = {entry["label"]: entry for entry in supplied}
    wanted = labels_in(text)
    for label in claimed or []:
        if label not in wanted:
            wanted.append(label)

    resolved: list[Citation] = []
    invalid: list[str] = []
    for label in wanted:
        entry = by_label.get(label)
        if entry is None:
            invalid.append(label)
            continue
        resolved.append(
            Citation(
                label=label,
                chunk_id=str(entry.get("chunk_id", "")),
                doc_id=str(entry.get("doc_id", "")),
                source_path=str(entry.get("source_path", "")),
            )
        )
    return tuple(resolved), tuple(invalid)


@dataclass
class AnswerBuilder:
    """Assembles an Answer from the model's JSON plus the supplied context."""

    question: str
    supplied: list[dict[str, Any]]
    tool_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    _warnings: list[str] = field(default_factory=list)

    def from_payload(self, payload: dict[str, Any]) -> Answer:
        text = str(payload.get("answer", "")).strip()
        claimed = [str(x) for x in (payload.get("cited_labels") or [])]
        citations, invalid = resolve_citations(text, self.supplied, claimed)

        contradictions = tuple(
            Contradiction(
                description=str(item.get("description", "")).strip(),
                labels=tuple(str(x) for x in (item.get("labels") or [])),
            )
            for item in (payload.get("contradictions") or [])
            if str(item.get("description", "")).strip()
        )

        confidence = str(payload.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        return Answer(
            question=self.question,
            text=text,
            citations=citations,
            contradictions=contradictions,
            abstained=bool(payload.get("abstained", False)),
            abstain_reason=str(payload.get("abstain_reason", "")).strip(),
            invalid_labels=invalid,
            confidence=confidence,
            steps=tuple(str(x) for x in (payload.get("steps") or [])),
            context_size=len(self.supplied),
            tool_calls=self.tool_calls,
            prompt_tokens=self.prompt_tokens,
            output_tokens=self.output_tokens,
        )

    def failure(self, message: str) -> Answer:
        """An answer that could not be produced at all.

        Returned rather than raised so that the evaluation harness can record
        a model outage as a result for that question instead of aborting a run
        of 43 questions on one HTTP error.
        """
        return Answer(
            question=self.question,
            text="",
            abstained=True,
            abstain_reason="the language model could not be reached",
            context_size=len(self.supplied),
            tool_calls=self.tool_calls,
            error=message,
        )
