"""Tests for the HTTP layer.

Every dependency is overridden, so the whole API runs against an in-process
Qdrant, a stub embedder and a scripted model. No cluster, no weights, no
network — which is the reason the expensive objects are provided through
`Depends` rather than created as module globals.

What is asserted here is HTTP behaviour: status codes, the response contract,
and that a domain error becomes the right status. The logic underneath is
tested in its own suites.
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from ragsys.api import deps
from ragsys.api.app import create_app
from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import IN_MEMORY, QdrantChunkStore
from ragsys.ingestion.dedupe import ContentIndex
from ragsys.ingestion.models import Chunk, stable_id
from ragsys.reasoning.agent import Agent
from ragsys.reasoning.llm import LLMResponse, LLMUnavailable
from ragsys.retrieval.retriever import Retriever

DOCUMENTS = [
    ("ops/services/ledger-service.json", "name: ledger-service team: Payments Core tier: 1"),
    ("ops/incidents/INC-2026-03-14.pdf", "Rollback complete at 15:10 after pool exhaustion."),
]


class StubEmbedder:
    dim = 8
    model_name = "stub-embedder"

    def __init__(self) -> None:
        self._model = object()

    @property
    def is_loaded(self) -> bool:
        return True

    def _vector(self, text: str) -> list[float]:
        angle = (hash(text) % 10_000) / 10_000 * 2 * math.pi
        raw = [math.cos(angle * (i + 1)) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class ScriptedClient:
    model = "scripted"

    def __init__(self, *responses) -> None:
        self.queue = list(responses)

    def generate(self, messages, system=None, tools=None, response_schema=None, temperature=0.0):
        nxt = self.queue.pop(0) if self.queue else self.queue
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def reply(payload: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload), finish_reason="STOP")


def build_store() -> QdrantChunkStore:
    stats = SparseStats.from_texts([text for _, text in DOCUMENTS])
    store = QdrantChunkStore(
        embedder=StubEmbedder(),
        sparse_encoder=SparseEncoder(stats),
        url=IN_MEMORY,
        collection="test_api",
    )
    store.ensure_collection(recreate=True)
    store.upsert(
        [
            Chunk(
                chunk_id=stable_id(path, str(i), "structural"),
                doc_id=path,
                text=text,
                index=i,
                strategy="structural",
                metadata={"source_path": path, "format": path.rsplit(".", 1)[-1]},
            )
            for i, (path, text) in enumerate(DOCUMENTS)
        ]
    )
    return store


def make_client(*scripted) -> TestClient:
    store = build_store()
    retriever = Retriever(store)
    agent = Agent(retriever, client=ScriptedClient(*scripted))
    index = ContentIndex()

    app = create_app(serve_static=False)
    app.dependency_overrides[deps.get_store] = lambda: store
    app.dependency_overrides[deps.get_retriever] = lambda: retriever
    app.dependency_overrides[deps.get_agent] = lambda: agent
    app.dependency_overrides[deps.get_content_index] = lambda: index
    return TestClient(app)


ANSWERED = reply(
    {
        "answer": "The ledger service is owned by the Payments Core team [S1].",
        "cited_labels": ["S1"],
        "confidence": "high",
        "steps": ["read the service catalogue"],
    }
)


# ---------- health ----------


def test_health_reports_the_index():
    response = make_client().get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == len(DOCUMENTS)
    assert body["embedding_model"] == "stub-embedder"
    assert body["model_loaded"] is True


def test_health_does_not_fail_when_the_store_is_unreachable():
    """A health check must report a problem, not become one."""
    client = make_client()
    store = client.app.dependency_overrides[deps.get_store]()

    def broken() -> int:
        raise ConnectionError("no route to host")

    store.count = broken
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert "could not be reached" in body["detail"]


# ---------- ask ----------


def test_ask_returns_an_answer_with_citations_and_sources():
    response = make_client(ANSWERED).post("/api/ask", json={"question": "who owns the ledger"})
    assert response.status_code == 200
    body = response.json()
    assert "Payments Core" in body["answer"]
    assert body["grounded"] is True
    assert [c["label"] for c in body["citations"]] == ["S1"]
    assert body["citations"][0]["source_path"]
    # Every retrieved source is returned, not only the cited ones, so the
    # interface can show what the model was given.
    assert len(body["sources"]) >= len(body["citations"])
    assert body["model"] == "scripted"


def test_ask_rejects_an_empty_question():
    response = make_client().post("/api/ask", json={"question": "x"})
    assert response.status_code == 422


def test_ask_requires_a_question_field():
    assert make_client().post("/api/ask", json={}).status_code == 422


def test_a_model_outage_is_reported_in_the_answer_not_as_a_crash():
    response = make_client(LLMUnavailable("HTTP 503")).post(
        "/api/ask", json={"question": "who owns the ledger"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["grounded"] is False


def test_an_invented_citation_is_surfaced():
    response = make_client(reply({"answer": "As shown [S99].", "cited_labels": ["S99"]})).post(
        "/api/ask", json={"question": "who owns the ledger"}
    )
    body = response.json()
    assert body["invalid_citations"] == ["S99"]
    assert body["grounded"] is False


# ---------- documents ----------


def test_listing_documents_groups_by_source_file():
    response = make_client().get("/api/documents")
    assert response.status_code == 200
    body = response.json()
    paths = {d["source_path"] for d in body["documents"]}
    assert paths == {path for path, _ in DOCUMENTS}
    assert body["total_chunks"] == len(DOCUMENTS)


def test_listing_respects_the_limit():
    body = make_client().get("/api/documents", params={"limit": 1}).json()
    assert body["shown"] == 1


def test_an_absurd_limit_is_rejected():
    assert make_client().get("/api/documents", params={"limit": 0}).status_code == 422


def test_uploading_a_document_indexes_it():
    client = make_client()
    response = client.post(
        "/api/documents",
        files={
            "file": (
                "notes.md",
                b"# Title\n\nThe kyc-service is owned by Compliance.\n",
                "text/markdown",
            )
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["chunks_written"] >= 1
    assert body["states"] == ["new"]
    assert any(
        d["source_path"].endswith("notes.md")
        for d in client.get("/api/documents").json()["documents"]
    )


def test_re_uploading_replaces_rather_than_duplicates():
    client = make_client()
    payload = {"file": ("notes.md", b"# One\n\nfirst version of the note\n", "text/markdown")}
    client.post("/api/documents", files=payload)
    before = client.get("/api/health").json()["chunks"]
    client.post(
        "/api/documents",
        files={"file": ("notes.md", b"# Two\n\na different second version\n", "text/markdown")},
    )
    after = client.get("/api/health").json()["chunks"]
    assert after == before


def test_re_uploading_an_identical_file_keeps_it_indexed():
    """The regression this guards lost the document entirely.

    Replacing deletes the file's chunks first. If change detection then
    compares the content against what it already recorded and reports
    UNCHANGED, the write is skipped and the delete is never undone.
    """
    client = make_client()
    payload = {"file": ("same.md", b"# Same\n\nidentical content every time\n", "text/markdown")}

    first = client.post("/api/documents", files=payload).json()
    assert first["chunks_written"] >= 1
    after_first = client.get("/api/health").json()["chunks"]

    second = client.post("/api/documents", files=payload).json()
    assert second["chunks_written"] >= 1, "an identical re-upload must still write"
    assert client.get("/api/health").json()["chunks"] == after_first

    assert any(
        d["source_path"].endswith("same.md")
        for d in client.get("/api/documents").json()["documents"]
    ), "the document disappeared after being re-uploaded"


def test_an_upload_is_chunked_with_one_strategy():
    """The live index holds one chunking; writing both doubles an upload.

    The offline build writes every strategy so they can be compared. An
    upload that did the same would store each passage twice under two
    chunkings, giving uploaded documents twice the result slots of documents
    that arrived in the bulk build.
    """
    client = make_client()
    client.post(
        "/api/documents",
        files={"file": ("one.md", b"# One\n\n" + b"word " * 400 + b"\n", "text/markdown")},
    )
    store = client.app.dependency_overrides[deps.get_store]()
    strategies = {
        c.payload["strategy"]
        for c in store.client.scroll(collection_name="test_api", limit=200, with_payload=True)[0]
        if c.payload.get("meta_source_path", "").endswith("one.md")
    }
    assert strategies == {"structural"}


def test_an_unsupported_file_type_is_rejected():
    response = make_client().post(
        "/api/documents", files={"file": ("photo.png", b"\x89PNG", "image/png")}
    )
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_file"


def test_a_filename_cannot_escape_the_upload_directory():
    """A browser can send any name; only the final component may be used."""
    response = make_client().post(
        "/api/documents",
        files={"file": ("../../../etc/passwd.md", b"# owned\n\nnope\n", "text/markdown")},
    )
    assert response.status_code == 201
    listed = make_client().get("/api/documents").json()
    assert not any(".." in d["source_path"] for d in listed["documents"])


def test_deleting_a_source_removes_its_chunks():
    client = make_client()
    target = DOCUMENTS[0][0]
    response = client.request("DELETE", "/api/documents", params={"source_path": target})
    assert response.status_code == 200
    assert response.json()["chunks_removed"] == 1
    assert client.get("/api/health").json()["chunks"] == len(DOCUMENTS) - 1


def test_deleting_something_absent_reports_zero():
    body = (
        make_client()
        .request("DELETE", "/api/documents", params={"source_path": "ops/nope.md"})
        .json()
    )
    assert body["chunks_removed"] == 0


# ---------- error mapping ----------


def test_a_missing_index_becomes_a_503():
    """The one place a domain error maps to a status code is the app factory."""
    app = create_app(serve_static=False)

    def unavailable():
        raise deps.IndexNotBuilt("sparse_stats_structural.json is missing")

    app.dependency_overrides[deps.get_store] = unavailable
    response = TestClient(app, raise_server_exceptions=False).get("/api/health")
    assert response.status_code == 503
    assert response.json()["error"] == "index_not_built"


@pytest.mark.parametrize("path", ["/api/health", "/api/documents"])
def test_the_openapi_schema_documents_every_endpoint(path):
    schema = make_client().get("/openapi.json").json()
    assert path in schema["paths"]
