"""The application: routers wired up, errors mapped, static files served.

Every domain error is translated to a status code here and nowhere else. A
try/except in each router would scatter the same mapping across files and
leave a new router without it; registered once, every endpoint inherits it.

Built by a factory rather than at import, so tests can construct an app with
dependencies overridden instead of inheriting whatever the module did on the
way in.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ragsys.api.deps import IndexNotBuilt
from ragsys.api.routers import ask, documents, health
from ragsys.indexing.vector_store import VectorStoreNotConfigured
from ragsys.ingestion.loaders import UnsupportedFormatError
from ragsys.ingestion.pipeline import UnsupportedUpload
from ragsys.logging import get_logger, setup_logging
from ragsys.reasoning.llm import LLMError, LLMNotConfigured

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

TITLE = "Document investigation system"
DESCRIPTION = (
    "Hybrid retrieval over a mixed-format document corpus, with evidence-grounded "
    "answers, inline citations, contradiction reporting and abstention."
)

# Which domain failures mean what over HTTP. 503 for "correctly built but a
# dependency is unavailable", 415 for "the request named something we cannot
# read". Nothing here is a 500: a 500 means we did not anticipate it.
ERROR_STATUS: tuple[tuple[type[Exception], int, str], ...] = (
    (UnsupportedUpload, 415, "unsupported_file"),
    (UnsupportedFormatError, 415, "unsupported_file"),
    (IndexNotBuilt, 503, "index_not_built"),
    (VectorStoreNotConfigured, 503, "vector_store_unavailable"),
    (LLMNotConfigured, 503, "model_not_configured"),
    (LLMError, 503, "model_unavailable"),
)


def _register_error_handlers(app: FastAPI) -> None:
    def make_handler(status: int, code: str):
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            log.warning("%s %s -> %s: %s", request.method, request.url.path, code, exc)
            return JSONResponse(status_code=status, content={"error": code, "detail": str(exc)})

        return handler

    # Registered most specific first: LLMNotConfigured subclasses LLMError, and
    # Starlette picks the handler for the exact class before a base class.
    for exception, status, code in ERROR_STATUS:
        app.add_exception_handler(exception, make_handler(status, code))


def create_app(serve_static: bool = True) -> FastAPI:
    setup_logging()
    app = FastAPI(title=TITLE, description=DESCRIPTION, version="1.0.0")

    app.include_router(ask.router)
    app.include_router(documents.router)
    app.include_router(health.router)
    _register_error_handlers(app)

    # Mounted last so that /api/* is matched by a router first. `html=True`
    # serves index.html at the root.
    if serve_static and STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
