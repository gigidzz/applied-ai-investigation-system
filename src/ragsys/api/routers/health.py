"""GET /api/health

Reports readiness without causing it. Touching the embedding model would load
130 MB of weights and block for about twenty-five seconds, which is long
enough for a hosting platform's health probe to conclude the container is
dead and restart it — turning the health check into the outage it exists to
detect. So the model's state is read, never triggered.
"""

from __future__ import annotations

from fastapi import APIRouter

from ragsys.api.deps import StoreDep
from ragsys.api.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Readiness and index size")
def health(store: StoreDep) -> HealthResponse:
    try:
        chunks = store.count()
    except Exception as exc:
        return HealthResponse(
            status="degraded",
            collection=store.collection,
            chunks=None,
            embedding_model=store.embedder.model_name,
            model_loaded=store.embedder.is_loaded,
            detail=f"the vector store could not be reached: {type(exc).__name__}",
        )
    return HealthResponse(
        status="ok" if chunks else "empty",
        collection=store.collection,
        chunks=chunks,
        embedding_model=store.embedder.model_name,
        model_loaded=store.embedder.is_loaded,
        detail=None if chunks else "the collection holds no chunks; run scripts/build_index.py",
    )
