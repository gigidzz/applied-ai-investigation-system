"""Qdrant-backed chunk storage.

This class satisfies the same ChunkStore protocol the ingestion pipeline
already talks to, which is why no ingestion code changes to put chunks in a
vector database instead of a file. That was the point of defining the protocol
before a database existed.

Each point carries two vectors under one id: a dense embedding for meaning and
a sparse term vector for exact wording. Storing them together means a chunk is
retrieved by either route without duplicating its text, and lets retrieval
merge two rankings over the same identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from ragsys.config import settings
from ragsys.indexing.embedder import Embedder
from ragsys.indexing.sparse import SparseEncoder
from ragsys.ingestion.models import Chunk
from ragsys.logging import get_logger

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"

# Document-level fields worth surfacing in a listing, and the payload key each
# comes from. Filled from whichever chunk happens to carry them.
_FILLABLE = (("date", "meta_date"), ("service", "meta_service"), ("title", "meta_title"))

# Qdrant serves HTTP on 6333. The cloud console sometimes hands out a URL
# without the port, and a missing port surfaces much later as an opaque
# connection error, so it is filled in here instead.
DEFAULT_PORT = 6333

# Sentinel for an in-process instance, used by the tests.
IN_MEMORY = ":memory:"


class VectorStoreNotConfigured(RuntimeError):
    """No Qdrant URL available, locally or in the environment."""


def point_id(chunk_id: str) -> int:
    """Qdrant addresses points by unsigned 64-bit integer or UUID.

    Chunk ids are 16 hexadecimal characters, which is exactly 64 bits, so the
    id converts without truncation and stays deterministic. Collision risk
    across this corpus is on the order of 1 in 2 billion.
    """
    return int(chunk_id, 16)


def normalise_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.port is None:
        parsed = parsed._replace(netloc=f"{parsed.hostname}:{DEFAULT_PORT}")
    return urlunparse(parsed)


@dataclass(frozen=True)
class DocumentSummary:
    """One source file as the interface presents it.

    Grouped by source path rather than by document id, because a user thinks
    in files while the index thinks in documents: a three-page PDF is three
    documents with ids ending `#page=1`, `#page=2`, `#page=3`, and offering
    three rows to delete separately would be confusing and error-prone.
    """

    source_path: str
    fmt: str
    chunk_count: int
    doc_ids: tuple[str, ...]
    date: str | None = None
    service: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, with everything a citation needs."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    payload: dict[str, Any]

    @property
    def source_path(self) -> str:
        return str(self.payload.get("meta_source_path", self.doc_id))


class QdrantChunkStore:
    """Dense and sparse vectors for every chunk, in one collection."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        timeout: int = 120,
    ) -> None:
        resolved = url or settings.qdrant_url
        if not resolved:
            raise VectorStoreNotConfigured(
                "Set QDRANT_URL in .env, or pass url= for a local instance."
            )
        key = api_key
        if key is None and settings.qdrant_api_key is not None:
            key = settings.qdrant_api_key.get_secret_value()

        from qdrant_client import QdrantClient

        self.collection = collection or settings.qdrant_collection
        self.embedder = embedder or Embedder()
        self.sparse_encoder = sparse_encoder
        # ":memory:" runs Qdrant in-process. The tests use it so that store
        # behaviour — upsert, delete, search — is verified against the real
        # client rather than a hand-written mock, without needing a cluster.
        self._in_memory = resolved == IN_MEMORY
        if self._in_memory:
            self.client = QdrantClient(location=IN_MEMORY)
        else:
            self.client = QdrantClient(url=normalise_url(resolved), api_key=key, timeout=timeout)

    # ---------- schema ----------

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Create the collection if absent. Returns True if it was created."""
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return False
        if exists:
            log.warning("deleting existing collection %s", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.embedder.dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams()},
        )
        # Deletion and filtering both key on doc_id, and an unindexed payload
        # field forces a full scan of every point to resolve either. The
        # in-process instance ignores payload indexes, so asking for one there
        # only produces a warning.
        if not self._in_memory:
            for field in ("doc_id", "strategy", "meta_format", "meta_source_path"):
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword",
                )
        log.info("created collection %s (dim=%d)", self.collection, self.embedder.dim)
        return True

    # ---------- ChunkStore protocol ----------

    def upsert(self, chunks: list[Chunk], wait: bool = True) -> int:
        """Embed and store a batch.

        Batch size matters in two different places and they do not agree:
        the embedder is fastest at 64 texts, while each upsert pays one
        round trip to the cluster regardless of size, so uploading 64 at a
        time spends as long waiting on the network as on the model. The
        caller therefore passes a larger batch here and the embedder
        subdivides it internally.
        """
        from qdrant_client import models

        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        points = []
        for chunk, dense in zip(chunks, vectors, strict=True):
            payload = chunk.to_record()
            vector: dict[str, Any] = {DENSE: dense}
            if self.sparse_encoder is not None:
                sparse = self.sparse_encoder.encode_document(chunk.text)
                if len(sparse):
                    vector[SPARSE] = models.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    )
            points.append(
                models.PointStruct(id=point_id(chunk.chunk_id), vector=vector, payload=payload)
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=wait)
        return len(points)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document.

        Counted before deleting because Qdrant reports no deleted total, and
        the protocol promises one. Two round trips is the cost of an honest
        return value.
        """
        from qdrant_client import models

        condition = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        present = self.client.count(
            collection_name=self.collection, count_filter=condition, exact=True
        ).count
        if present:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=condition),
                wait=True,
            )
        return present

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def ensure_payload_indexes(self) -> list[str]:
        """Add any missing payload indexes to an existing collection.

        Needed because the collection was created before document listing
        existed, and filtering an unindexed payload field forces a scan of
        every point. Creating an index that already exists is harmless, so
        this is safe to call on every startup.
        """
        if self._in_memory:
            return []
        created = []
        for field in ("doc_id", "strategy", "meta_format", "meta_source_path"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema="keyword"
                )
                created.append(field)
            except Exception as exc:  # already present, which is the common case
                log.debug("payload index %s not created: %s", field, exc)
        return created

    def list_documents(
        self,
        limit: int = 500,
        exclude_formats: tuple[str, ...] = ("jsonl",),
    ) -> list[DocumentSummary]:
        """Source files in the index, newest first, excluding bulk corpora.

        `jsonl` is excluded by default because that format holds the 66,581
        Wikipedia paragraphs: they arrived as one bulk dataset, are not
        individually manageable, and listing them would bury the documents a
        user can actually act on. Only a few hundred points match once they
        are filtered out, so grouping client-side costs one or two round trips
        rather than a scan of the whole collection.
        """
        from qdrant_client import models

        condition = None
        if exclude_formats:
            condition = models.Filter(
                must_not=[
                    models.FieldCondition(key="meta_format", match=models.MatchValue(value=fmt))
                    for fmt in exclude_formats
                ]
            )

        by_source: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=condition,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                source = str(payload.get("meta_source_path") or payload.get("doc_id") or "")
                if not source:
                    continue
                entry = by_source.setdefault(
                    source,
                    {
                        "fmt": str(payload.get("meta_format", "")),
                        "chunks": 0,
                        "doc_ids": set(),
                        "date": payload.get("meta_date"),
                        "service": payload.get("meta_service"),
                        "title": payload.get("meta_title"),
                    },
                )
                entry["chunks"] += 1
                entry["doc_ids"].add(str(payload.get("doc_id", "")))
                # The first chunk of a document often lacks a date while a
                # later one carries it, so fill gaps as they appear.
                for key, meta in _FILLABLE:
                    if entry[key] is None and payload.get(meta) is not None:
                        entry[key] = payload[meta]
            if offset is None or len(by_source) >= limit:
                break

        summaries = [
            DocumentSummary(
                source_path=source,
                fmt=entry["fmt"],
                chunk_count=entry["chunks"],
                doc_ids=tuple(sorted(d for d in entry["doc_ids"] if d)),
                date=entry["date"],
                service=entry["service"],
                title=entry["title"],
            )
            for source, entry in by_source.items()
        ]
        # Dated documents first and newest first, because a reader scanning
        # an operations corpus wants recent incidents at the top. Undated
        # documents follow in path order, which is at least predictable.
        dated = sorted((d for d in summaries if d.date), key=lambda d: str(d.date), reverse=True)
        undated = sorted((d for d in summaries if not d.date), key=lambda d: d.source_path)
        return (dated + undated)[:limit]

    def delete_source(self, source_path: str) -> int:
        """Remove every chunk of a source file, across all its documents.

        The file-level counterpart to delete_document: deleting a three-page
        PDF should not require three calls.
        """
        from qdrant_client import models

        condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="meta_source_path", match=models.MatchValue(value=source_path)
                )
            ]
        )
        present = self.client.count(
            collection_name=self.collection, count_filter=condition, exact=True
        ).count
        if present:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=condition),
                wait=True,
            )
        return present

    # ---------- retrieval ----------

    def search_dense(self, query: str, limit: int = 20) -> list[Hit]:
        """Nearest neighbours by meaning."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.embed_query(query),
            using=DENSE,
            limit=limit,
            with_payload=True,
        )
        return [_to_hit(p) for p in response.points]

    def search_sparse(self, query: str, limit: int = 20) -> list[Hit]:
        """Best matches by weighted term overlap, which is BM25 here."""
        from qdrant_client import models

        if self.sparse_encoder is None:
            raise VectorStoreNotConfigured("sparse search needs a SparseEncoder")
        vector = self.sparse_encoder.encode_query(query)
        if not len(vector):
            return []
        response = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE,
            limit=limit,
            with_payload=True,
        )
        return [_to_hit(p) for p in response.points]


def _to_hit(point: Any) -> Hit:
    payload = point.payload or {}
    return Hit(
        chunk_id=str(payload.get("chunk_id", point.id)),
        doc_id=str(payload.get("doc_id", "")),
        text=str(payload.get("text", "")),
        score=float(point.score),
        payload=payload,
    )
