"""The shared objects, created once.

The embedding model costs roughly twenty-five seconds and 130 MB to load, and
the vector store holds a connection. Building either per request would make
every question unusable, so they are built once and handed to the routers.

`lru_cache` rather than module-level globals, for one reason that matters: a
test can replace any of these through FastAPI's dependency overrides and run
the whole API against an in-process Qdrant and a stub embedder, with no
cluster, no model and no network. A global cannot be swapped that cleanly.

This is also where the retrieval configuration is chosen. A router should not
know that `hybrid+rerank` exists, and the domain should not know it is the
default; naming it once here is the compromise.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ragsys.config import PROJECT_ROOT, settings
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import QdrantChunkStore
from ragsys.ingestion.dedupe import ContentIndex
from ragsys.logging import get_logger
from ragsys.reasoning.agent import Agent
from ragsys.retrieval.retriever import HYBRID_RERANKED, Retriever

log = get_logger(__name__)


class IndexNotBuilt(RuntimeError):
    """The sparse statistics are missing, so the index was never built here."""


def sparse_stats_path():
    return PROJECT_ROOT / "data" / "processed" / f"sparse_stats_{settings.chunk_strategy}.json"


@lru_cache(maxsize=1)
def get_store() -> QdrantChunkStore:
    path = sparse_stats_path()
    if not path.exists():
        raise IndexNotBuilt(
            f"{path.name} is missing. Run scripts/build_index.py to create the index, "
            "or ship the statistics file alongside the application."
        )
    store = QdrantChunkStore(
        embedder=Embedder(),
        sparse_encoder=SparseEncoder(SparseStats.load(path)),
    )
    # Cheap, idempotent, and it repairs a collection built before document
    # listing existed. Filtering an unindexed payload field would otherwise
    # scan every one of the 72,294 points.
    store.ensure_payload_indexes()
    return store


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(get_store(), config=HYBRID_RERANKED)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    return Agent(get_retriever())


@lru_cache(maxsize=1)
def get_content_index() -> ContentIndex:
    """Change detection state for this process.

    Deliberately empty at startup rather than rebuilt from 72,294 stored
    chunks, which would add minutes to boot for a benefit uploads do not
    need: an upload replaces its file's chunks outright, so correctness does
    not depend on remembering what was ingested before this process started.
    The index still reports accurately within a session.
    """
    return ContentIndex()


# Annotated aliases, which is the current FastAPI style and keeps the routers
# readable: a signature says `store: StoreDep` rather than repeating
# `Depends(get_store)` at every call site. It also satisfies the rule against
# calling a function in an argument default, which the older style violates.
StoreDep = Annotated[QdrantChunkStore, Depends(get_store)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
AgentDep = Annotated[Agent, Depends(get_agent)]
ContentIndexDep = Annotated[ContentIndex, Depends(get_content_index)]
