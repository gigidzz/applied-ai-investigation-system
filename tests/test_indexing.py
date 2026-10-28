"""Tests for the dense and sparse halves of the index.

The sparse encoder is checked against an independent BM25 implementation
rather than against stored expectations, because the whole design rests on one
claim — that a dot product of the two sparse vectors reproduces BM25 — and a
hand-written expected value would only restate the implementation.

Store behaviour runs against an in-process Qdrant rather than a mock, so
upsert, delete and search are exercised through the real client.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from ragsys.indexing.sparse import (
    K1,
    B,
    SparseEncoder,
    SparseStats,
    tokenise,
)
from ragsys.indexing.vector_store import (
    IN_MEMORY,
    QdrantChunkStore,
    normalise_url,
    point_id,
)
from ragsys.ingestion.models import Chunk, stable_id

CORPUS = [
    "payment gateway returned 503 errors after pay-gw-4.2.1 was deployed",
    "the ledger service runbook describes the nightly reconciliation job",
    "card authorisation was failing during the payment gateway incident",
    "kyc service owns identity verification and document checks",
    "connection pool exhaustion caused the payment gateway outage",
]


@pytest.fixture(scope="module")
def stats() -> SparseStats:
    return SparseStats.from_texts(CORPUS)


@pytest.fixture(scope="module")
def encoder(stats: SparseStats) -> SparseEncoder:
    return SparseEncoder(stats)


# ---------- tokenisation ----------


def test_compound_identifier_is_kept_whole():
    assert "pay-gw-4.2.1" in tokenise("rollback of pay-gw-4.2.1 complete")


def test_compound_identifier_is_also_split():
    tokens = tokenise("INC-2026-03-14 hit payment-gateway")
    # Both forms present: the whole identifier for an exact query, the parts so
    # that "payment gateway" matches a document writing "payment-gateway".
    assert "inc-2026-03-14" in tokens
    assert {"inc", "2026", "payment", "gateway"} <= set(tokens)


def test_tokenisation_is_case_insensitive():
    assert tokenise("Payment GATEWAY") == tokenise("payment gateway")


# ---------- BM25 ----------


def reference_bm25(query: str, document: str, stats: SparseStats) -> float:
    """BM25 written out directly, as an independent check."""
    tokens = tokenise(document)
    frequencies = Counter(tokens)
    length = len(tokens)
    score = 0.0
    for term in dict.fromkeys(tokenise(query)):
        if term not in frequencies:
            continue
        idf = stats.idf(term)
        if idf <= 0.0:
            continue
        tf = frequencies[term]
        score += idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * length / stats.average_length))
    return score


@pytest.mark.parametrize(
    "query",
    [
        "payment gateway pool exhaustion",
        "pay-gw-4.2.1",
        "ledger reconciliation",
        "kyc identity verification",
    ],
)
def test_sparse_dot_product_equals_bm25(query, encoder, stats):
    """The claim the whole hybrid design rests on."""
    query_vector = encoder.encode_query(query)
    weights = dict(zip(query_vector.indices, query_vector.values, strict=True))
    for document in CORPUS:
        document_vector = encoder.encode_document(document)
        dot = sum(
            value * weights.get(index, 0.0)
            for index, value in zip(document_vector.indices, document_vector.values, strict=True)
        )
        assert dot == pytest.approx(reference_bm25(query, document, stats), abs=1e-9)


def test_idf_never_negative(stats):
    """A term in every document must not drag a score below one that omits it."""
    for term in stats.document_frequency:
        assert stats.idf(term) >= 0.0


def test_idf_rewards_rare_terms(stats):
    assert stats.idf("exhaustion") > stats.idf("payment")


def test_unknown_query_terms_are_dropped(encoder):
    vector = encoder.encode_query("zzzznotacorpusword")
    assert len(vector) == 0


def test_document_frequency_counts_documents_not_occurrences():
    stats = SparseStats.from_texts(["alpha alpha alpha", "alpha beta"])
    assert stats.document_frequency["alpha"] == 2
    assert stats.document_frequency["beta"] == 1


def test_stats_survive_a_round_trip(tmp_path, stats):
    path = tmp_path / "stats.json"
    stats.save(path)
    reloaded = SparseStats.load(path)
    assert reloaded.total_documents == stats.total_documents
    assert reloaded.vocabulary == stats.vocabulary
    assert reloaded.idf("payment") == pytest.approx(stats.idf("payment"))


def test_empty_text_encodes_to_nothing(encoder):
    assert len(encoder.encode_document("")) == 0


# ---------- identifiers and urls ----------


def test_point_id_is_deterministic_and_fits_64_bits():
    chunk_id = stable_id("some/doc.pdf", "3", "structural")
    assert point_id(chunk_id) == point_id(chunk_id)
    assert 0 <= point_id(chunk_id) < 2**64


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://host.cloud.qdrant.io", "https://host.cloud.qdrant.io:6333"),
        ("https://host.cloud.qdrant.io:6333", "https://host.cloud.qdrant.io:6333"),
        ("https://host.cloud.qdrant.io:443", "https://host.cloud.qdrant.io:443"),
        ("host.cloud.qdrant.io", "https://host.cloud.qdrant.io:6333"),
    ],
)
def test_missing_port_is_filled_in(given, expected):
    assert normalise_url(given) == expected


# ---------- store behaviour ----------


class StubEmbedder:
    """Deterministic vectors from text, so nearest-neighbour order is knowable.

    Each text maps to a point on a circle by hashing. Querying the exact text
    of a stored chunk therefore returns that chunk first, which is what makes
    the search assertions meaningful without loading a real model.
    """

    dim = 8

    def _vector(self, text: str) -> list[float]:
        angle = (hash(text) % 10_000) / 10_000 * 2 * math.pi
        raw = [math.cos(angle * (i + 1)) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_chunks() -> list[Chunk]:
    chunks = []
    for index, text in enumerate(CORPUS):
        doc_id = f"ops/doc-{index}.md"
        chunks.append(
            Chunk(
                chunk_id=stable_id(doc_id, str(index), "structural"),
                doc_id=doc_id,
                text=text,
                index=index,
                strategy="structural",
                metadata={"source_path": doc_id, "format": "md"},
            )
        )
    return chunks


@pytest.fixture
def store(encoder) -> QdrantChunkStore:
    store = QdrantChunkStore(
        embedder=StubEmbedder(),
        sparse_encoder=encoder,
        url=IN_MEMORY,
        collection="test_chunks",
    )
    store.ensure_collection(recreate=True)
    return store


def test_upsert_then_count(store):
    assert store.upsert(make_chunks()) == len(CORPUS)
    assert store.count() == len(CORPUS)


def test_upsert_is_idempotent(store):
    """Re-running the index build must not duplicate points."""
    chunks = make_chunks()
    store.upsert(chunks)
    store.upsert(chunks)
    assert store.count() == len(CORPUS)


def test_delete_removes_every_chunk_of_a_document(store):
    store.upsert(make_chunks())
    removed = store.delete_document("ops/doc-2.md")
    assert removed == 1
    assert store.count() == len(CORPUS) - 1


def test_delete_reports_zero_for_an_unknown_document(store):
    store.upsert(make_chunks())
    assert store.delete_document("ops/never-existed.md") == 0
    assert store.count() == len(CORPUS)


def test_dense_search_returns_payload_and_text(store):
    store.upsert(make_chunks())
    hits = store.search_dense(CORPUS[4], limit=3)
    assert hits
    assert hits[0].text == CORPUS[4]
    assert hits[0].doc_id == "ops/doc-4.md"
    assert hits[0].source_path == "ops/doc-4.md"


def test_sparse_search_finds_an_exact_identifier(store):
    """The case dense retrieval is known to be weak at."""
    store.upsert(make_chunks())
    hits = store.search_sparse("pay-gw-4.2.1", limit=3)
    assert hits
    assert hits[0].doc_id == "ops/doc-0.md"


def test_sparse_search_with_no_known_terms_returns_nothing(store):
    store.upsert(make_chunks())
    assert store.search_sparse("zzzznotacorpusword") == []


def test_upsert_of_nothing_is_harmless(store):
    assert store.upsert([]) == 0
    assert store.count() == 0
