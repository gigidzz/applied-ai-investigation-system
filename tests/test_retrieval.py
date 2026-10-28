"""Tests for fusion, the ranking signals, and selection.

The fusion test asserts the property the design depends on — that agreement
between two routes outranks a first place in either one alone — rather than
stored scores, which would only restate the formula.

Selection is tested through a real in-process Qdrant, so the crowding penalty
is exercised against the same client the deployed system uses.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from ragsys.indexing.vector_store import IN_MEMORY, Hit, QdrantChunkStore
from ragsys.ingestion.models import Chunk, stable_id
from ragsys.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from ragsys.retrieval.retriever import (
    DENSE_ONLY,
    HYBRID_FUSION_ONLY,
    HYBRID_RERANKED,
    RetrievalConfig,
    Retriever,
)
from ragsys.retrieval.signals import (
    QuestionKind,
    RerankWeights,
    classify_question,
    document_kind,
    metadata_agreement,
    query_entities,
    question_names_a_service,
    recency,
    type_preference,
)


def hit(chunk_id: str, doc_id: str = "d", text: str = "t", **payload) -> Hit:
    full = {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, **payload}
    return Hit(chunk_id=chunk_id, doc_id=doc_id, text=text, score=0.0, payload=full)


# ---------- fusion ----------


def test_agreement_outranks_a_single_first_place():
    """The measured case: dense ranked the right release third, sparse first."""
    rankings = {
        "dense": [hit("wrong-a"), hit("wrong-b"), hit("right")],
        "sparse": [hit("right"), hit("other-a"), hit("other-b")],
    }
    fused = reciprocal_rank_fusion(rankings)
    assert fused[0].chunk_id == "right"
    assert fused[0].agreed
    assert fused[0].fusion_score > fused[1].fusion_score


def test_fusion_score_is_the_sum_of_reciprocal_ranks():
    rankings = {"a": [hit("x"), hit("y")], "b": [hit("y")]}
    fused = {c.chunk_id: c for c in reciprocal_rank_fusion(rankings)}
    k = DEFAULT_RRF_K
    assert fused["x"].fusion_score == pytest.approx(1 / (k + 1))
    assert fused["y"].fusion_score == pytest.approx(1 / (k + 2) + 1 / (k + 1))


def test_routes_are_recorded_for_explanation():
    rankings = {"dense": [hit("x")], "sparse": [hit("x")]}
    candidate = reciprocal_rank_fusion(rankings)[0]
    assert candidate.routes == ("dense", "sparse")
    assert candidate.ranks == {"dense": 1, "sparse": 1}


def test_a_chunk_in_one_route_only_is_not_marked_agreed():
    candidate = reciprocal_rank_fusion({"dense": [hit("x")]})[0]
    assert not candidate.agreed


def test_route_weights_can_favour_one_route():
    rankings = {"dense": [hit("d")], "sparse": [hit("s")]}
    fused = reciprocal_rank_fusion(rankings, weights={"sparse": 3.0})
    assert fused[0].chunk_id == "s"


def test_fusing_nothing_returns_nothing():
    assert reciprocal_rank_fusion({"dense": []}) == []


# ---------- question classification ----------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("when did the rollback finish", QuestionKind.TEMPORAL),
        ("what time did the incident start", QuestionKind.TEMPORAL),
        ("how long did the outage last", QuestionKind.TEMPORAL),
        ("how do I restart the payment gateway", QuestionKind.PROCEDURAL),
        ("what steps mitigate pool exhaustion", QuestionKind.PROCEDURAL),
        ("which services depend on the payment gateway", QuestionKind.DEPENDENCY),
        ("who owns the ledger service", QuestionKind.DEPENDENCY),
        ("what is the connection pool minimum", QuestionKind.FACTUAL),
    ],
)
def test_question_classification(question, expected):
    assert classify_question(question) == expected


def test_temporal_wins_over_procedural_when_both_could_match():
    """ "How long did X take" starts with "how" but is asking for a time."""
    assert classify_question("how long did the rollback take") == QuestionKind.TEMPORAL


# ---------- document kinds and type preference ----------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("ops/incidents/INC-2026-03-14.pdf", "incidents"),
        ("ops/runbooks/runbook-rollback.md", "runbooks"),
        ("ops/threads/THREAD-2026-03-14.txt", "threads"),
        ("hotpot/paragraphs.jsonl", "wiki"),
        ("", "wiki"),
    ],
)
def test_document_kind_from_path(path, expected):
    assert document_kind({"meta_source_path": path}) == expected


def test_runbooks_are_demoted_for_temporal_questions():
    """The failure this signal exists for."""
    runbook = {"meta_source_path": "ops/runbooks/runbook-rollback.md"}
    incident = {"meta_source_path": "ops/incidents/INC-2026-03-14.pdf"}
    assert type_preference(QuestionKind.TEMPORAL, runbook) < 0
    assert type_preference(QuestionKind.TEMPORAL, incident) > 0


def test_runbooks_are_promoted_for_procedural_questions():
    runbook = {"meta_source_path": "ops/runbooks/runbook-rollback.md"}
    assert type_preference(QuestionKind.PROCEDURAL, runbook) > 0


def test_factual_questions_prefer_no_document_type():
    for path in ("ops/incidents/x.pdf", "ops/runbooks/y.md", "hotpot/paragraphs.jsonl"):
        assert type_preference(QuestionKind.FACTUAL, {"meta_source_path": path}) == 0.0


# ---------- metadata agreement ----------


def test_exact_release_match_scores_full_agreement():
    payload = {"meta_releases": ["pay-gw-4.2.1"]}
    assert metadata_agreement("when was pay-gw-4.2.1 deployed", payload) == 1.0


def test_wrong_release_scores_no_agreement():
    payload = {"meta_releases": ["pay-gw-4.6.0"]}
    assert metadata_agreement("when was pay-gw-4.2.1 deployed", payload) == 0.0


def test_incident_reference_matches():
    payload = {"meta_incident_refs": ["INC-2026-03-14"]}
    assert metadata_agreement("what happened in INC-2026-03-14", payload) == 1.0


def test_a_date_inside_an_incident_reference_is_not_a_separate_expectation():
    """INC-2026-03-14 contains 2026-03-14; counting both diluted an exact match."""
    payload = {"meta_incident_refs": ["INC-2026-03-14"]}
    assert metadata_agreement("what happened in INC-2026-03-14", payload) == 1.0
    assert "2026-03-14" not in query_entities("what caused INC-2026-03-14").dates


def test_a_bare_date_in_a_question_is_still_read():
    assert "2026-03-14" in query_entities("what shipped on 2026-03-14").dates


def test_service_matches_both_spellings():
    payload = {"meta_service": "payment-gateway"}
    assert metadata_agreement("the payment gateway failed", payload) > 0
    assert metadata_agreement("payment-gateway failed", payload) > 0


def test_service_does_not_match_an_unrelated_question(stats_free=None):
    payload = {"meta_service": "kyc-service"}
    assert metadata_agreement("the payment gateway failed", payload) == 0.0


def test_an_unmentioned_service_does_not_dilute_an_exact_release_match():
    """The chunk declares a service the question never names: not a miss."""
    payload = {"meta_releases": ["pay-gw-4.2.1"], "meta_service": "payment-gateway"}
    question = "when did the rollback of pay-gw-4.2.1 finish"
    assert metadata_agreement(question, payload) == 1.0


def test_the_wrong_service_is_still_a_miss():
    """When the question does name a service, a different one counts against."""
    payload = {"meta_service": "kyc-service"}
    assert metadata_agreement("what does payment-gateway depend on", payload) == 0.0


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what does payment-gateway depend on", True),
        ("who owns the ledger-service", True),
        ("when did the rollback of pay-gw-4.2.1 finish", False),
        ("what caused INC-2026-03-14", False),
        ("what superseded ADR-0007", False),
        ("why did it break", False),
    ],
)
def test_detecting_whether_a_question_names_a_service(question, expected):
    """Releases and incident references are service-shaped and must not count."""
    assert question_names_a_service(question) is expected


def test_no_identifiers_anywhere_means_no_signal():
    assert metadata_agreement("why did it break", {}) == 0.0


# ---------- recency ----------


def test_missing_date_is_neutral_not_penalised():
    """66,581 Wikipedia paragraphs carry no date and must not be suppressed."""
    assert recency({}) == 0.0


def test_unparseable_date_is_neutral():
    assert recency({"meta_date": "not-a-date"}) == 0.0


def test_recent_beats_old():
    anchor = date(2026, 6, 1)
    fresh = recency({"meta_date": "2026-05-01"}, reference=anchor)
    stale = recency({"meta_date": "2026-01-01"}, reference=anchor)
    assert 0.0 < stale < fresh <= 1.0


def test_beyond_the_window_scores_zero_but_never_negative():
    assert recency({"meta_date": "2000-01-01"}, reference=date(2026, 6, 1)) == 0.0


def test_weights_can_be_turned_off():
    off = RerankWeights.off()
    assert (off.metadata, off.doc_type, off.recency, off.crowding) == (0.0, 0.0, 0.0, 0.0)


# ---------- selection, against a real in-process store ----------


class StubEmbedder:
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


@pytest.fixture
def retriever() -> Retriever:
    store = QdrantChunkStore(embedder=StubEmbedder(), url=IN_MEMORY, collection="test_retrieval")
    store.ensure_collection(recreate=True)

    chunks = []
    # One verbose document that would otherwise monopolise the results.
    for i in range(8):
        doc = "ops/reports/ops-2026-03.md"
        chunks.append(
            Chunk(
                chunk_id=stable_id(doc, str(i), "structural"),
                doc_id=doc,
                text=f"rollback discussion part {i} pool exhaustion",
                index=i,
                strategy="structural",
                metadata={"source_path": doc, "format": "md"},
            )
        )
    # Distinct documents that should get a chance at the result set.
    for i, doc in enumerate(["ops/incidents/INC-2026-03-14.pdf", "ops/threads/THREAD-1.txt"]):
        chunks.append(
            Chunk(
                chunk_id=stable_id(doc, str(i), "structural"),
                doc_id=doc,
                text="rollback complete at 15:10 pool exhaustion",
                index=i,
                strategy="structural",
                metadata={"source_path": doc, "format": "pdf", "date": "2026-03-14"},
            )
        )
    store.upsert(chunks)
    return Retriever(store)


def test_final_k_is_respected(retriever):
    config = RetrievalConfig(name="t", use_sparse=False, final_k=4)
    result = retriever.retrieve("rollback", config)
    assert len(result.candidates) == 4


def test_crowding_increases_document_variety(retriever):
    """The verbose document must not fill every slot."""
    plain = HYBRID_FUSION_ONLY.variant("plain", use_sparse=False, final_k=6)
    spread = DENSE_ONLY.variant(
        "spread", use_sparse=False, final_k=6, weights=RerankWeights(crowding=0.5)
    )
    without = len(set(retriever.retrieve("rollback", plain).documents))
    with_penalty = len(set(retriever.retrieve("rollback", spread).documents))
    assert with_penalty >= without


def test_dense_only_runs_one_route(retriever):
    result = retriever.retrieve("rollback", DENSE_ONLY)
    assert set(result.route_counts) == {"dense"}


def test_a_configuration_with_no_routes_is_rejected(retriever):
    config = RetrievalConfig(name="broken", use_dense=False, use_sparse=False)
    with pytest.raises(ValueError):
        retriever.retrieve("rollback", config)


def test_context_is_labelled_for_citation(retriever):
    config = HYBRID_RERANKED.variant("t", use_sparse=False, final_k=3)
    context = retriever.retrieve("rollback", config).as_context()
    assert [entry["label"] for entry in context] == ["S1", "S2", "S3"]
    assert all(entry["source_path"] for entry in context)
    assert all("kind" in entry for entry in context)


def test_signals_are_recorded_on_every_returned_candidate(retriever):
    config = HYBRID_RERANKED.variant("t", use_sparse=False, final_k=5)
    result = retriever.retrieve("when did the rollback finish", config)
    for candidate in result.candidates:
        assert {"metadata", "doc_type", "recency", "multiplier", "crowding"} <= set(
            candidate.signals
        )


def test_question_kind_is_reported(retriever):
    result = retriever.retrieve("when did the rollback finish", DENSE_ONLY)
    assert result.kind == QuestionKind.TEMPORAL
