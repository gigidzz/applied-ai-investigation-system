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
            for field in ("doc_id", "strategy"):
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
