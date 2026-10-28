"""The published HTTP contract.

Separate from the domain objects on purpose. `Answer` and `DocumentSummary`
are internal shapes that change as the system changes; these are what callers
depend on. The duplication is the point: renaming a domain field must not
break a client, and an internal field must not be serialised to the world by
accident.

Each response owns its own mapping in a `from_domain` classmethod, so a router
never assembles a payload field by field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ragsys.indexing.vector_store import DocumentSummary
from ragsys.reasoning.answer import Answer


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SourceOut(BaseModel):
    """One retrieved chunk, as the interface shows it."""

    label: str
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    score: float
    kind: str | None = None
    date: str | None = None
    service: str | None = None

    @classmethod
    def from_context(cls, entry: dict[str, Any]) -> SourceOut:
        return cls(
            label=str(entry.get("label", "")),
            chunk_id=str(entry.get("chunk_id", "")),
            doc_id=str(entry.get("doc_id", "")),
            source_path=str(entry.get("source_path", "")),
            text=str(entry.get("text", "")),
            score=float(entry.get("score", 0.0)),
            kind=entry.get("kind"),
            date=entry.get("date"),
            service=entry.get("service"),
        )


class CitationOut(BaseModel):
    label: str
    chunk_id: str
    doc_id: str
    source_path: str


class ContradictionOut(BaseModel):
    description: str
    labels: list[str]


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    abstain_reason: str
    grounded: bool
    confidence: str
    citations: list[CitationOut]
    contradictions: list[ContradictionOut]
    sources: list[SourceOut]
    steps: list[str]
    invalid_citations: list[str]
    model: str
    tool_calls: int
    prompt_tokens: int
    output_tokens: int

    @classmethod
    def from_domain(cls, answer: Answer) -> AskResponse:
        return cls(
            question=answer.question,
            answer=answer.text,
            abstained=answer.abstained,
            abstain_reason=answer.abstain_reason,
            grounded=answer.is_grounded,
            confidence=answer.confidence,
            citations=[
                CitationOut(
                    label=c.label,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source_path=c.source_path,
                )
                for c in answer.citations
            ],
            contradictions=[
                ContradictionOut(description=c.description, labels=list(c.labels))
                for c in answer.contradictions
            ],
            sources=[SourceOut.from_context(s) for s in answer.sources],
            steps=list(answer.steps),
            invalid_citations=list(answer.invalid_labels),
            model=answer.model,
            tool_calls=answer.tool_calls,
            prompt_tokens=answer.prompt_tokens,
            output_tokens=answer.output_tokens,
        )


class DocumentOut(BaseModel):
    source_path: str
    format: str
    chunk_count: int
    date: str | None = None
    service: str | None = None
    title: str | None = None

    @classmethod
    def from_domain(cls, summary: DocumentSummary) -> DocumentOut:
        return cls(
            source_path=summary.source_path,
            format=summary.fmt,
            chunk_count=summary.chunk_count,
            date=summary.date,
            service=summary.service,
            title=summary.title,
        )


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    shown: int
    total_chunks: int


class UploadResponse(BaseModel):
    filename: str
    documents: int
    chunks_written: int
    # One of new / unchanged / changed / duplicate per document produced, so a
    # re-upload visibly reports "changed" rather than silently succeeding.
    states: list[str]


class DeleteResponse(BaseModel):
    source_path: str
    chunks_removed: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    chunks: int | None = None
    embedding_model: str
    model_loaded: bool
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
