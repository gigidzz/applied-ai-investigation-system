"""GET, POST and DELETE /api/documents

The upload endpoint reads a stream and hands the bytes to the domain. The
orchestration it looks like it should own — validating the extension, writing
the file, replacing any previous version — lives in `ingest_upload`, because a
command-line importer would want exactly the same behaviour and because that
function is tested without starting a server.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from ragsys.api.deps import ContentIndexDep, StoreDep
from ragsys.api.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentOut,
    UploadResponse,
)
from ragsys.ingestion.pipeline import ingest_upload

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse, summary="List source files")
def list_documents(
    store: StoreDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> DocumentListResponse:
    summaries = store.list_documents(limit=limit)
    return DocumentListResponse(
        documents=[DocumentOut.from_domain(s) for s in summaries],
        shown=len(summaries),
        total_chunks=store.count(),
    )


@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload and index a document",
)
async def upload_document(
    store: StoreDep,
    index: ContentIndexDep,
    file: Annotated[UploadFile, File()],
) -> UploadResponse:
    # Read before handing off: the domain function takes bytes, so that it has
    # no opinion about where an upload came from.
    data = await file.read()
    results = ingest_upload(file.filename or "", data, store, index)
    return UploadResponse(
        filename=file.filename or "",
        documents=len(results),
        chunks_written=sum(r.chunks_written for r in results),
        states=sorted({r.state.value for r in results}),
    )


@router.delete("/documents", response_model=DeleteResponse, summary="Delete a source file")
def delete_document(
    store: StoreDep,
    source_path: Annotated[str, Query(min_length=1)],
) -> DeleteResponse:
    removed = store.delete_source(source_path)
    return DeleteResponse(source_path=source_path, chunks_removed=removed)
