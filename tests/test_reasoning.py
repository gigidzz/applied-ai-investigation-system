"""Tests for the reasoning stage.

The agent is driven by a scripted fake client, so the loop, the tool round,
the fallbacks and the failure paths are all exercised without a network call
and without spending free-tier quota. What the tests assert is behaviour the
requirements name: that a citation the model invents is caught, that an
abstention survives, that disagreement is reported, and that a model outage
becomes a recorded result rather than an exception.
"""

from __future__ import annotations

import json
import math

import pytest

from ragsys.indexing.sparse import SparseEncoder, SparseStats
from ragsys.indexing.vector_store import IN_MEMORY, QdrantChunkStore
from ragsys.ingestion.models import Chunk, stable_id
from ragsys.reasoning.agent import Agent, AgentConfig, UnparseableAnswer, extract_json
from ragsys.reasoning.answer import (
    Answer,
    AnswerBuilder,
    labels_in,
    resolve_citations,
)
from ragsys.reasoning.llm import (
    RETRYABLE_STATUS,
    GeminiClient,
    LLMNotConfigured,
    LLMRefused,
    LLMResponse,
    LLMUnavailable,
    ToolCall,
)
from ragsys.reasoning.prompts import format_sources
from ragsys.reasoning.tools import SEARCH_TOOL_NAME, SourceRegister
from ragsys.retrieval.retriever import Retriever

SUPPLIED = [
    {
        "label": "S1",
        "chunk_id": "aaa",
        "doc_id": "ops/incidents/INC-1.pdf",
        "source_path": "ops/incidents/INC-1.pdf",
        "text": "Rollback complete at 15:10.",
    },
    {
        "label": "S2",
        "chunk_id": "bbb",
        "doc_id": "ops/threads/T-1.txt",
        "source_path": "ops/threads/T-1.txt",
        "text": "rollback finished about 15:40",
    },
]


# ---------- JSON extraction ----------


def test_plain_json_is_parsed():
    assert extract_json('{"answer": "x"}') == {"answer": "x"}


def test_fenced_json_is_parsed():
    assert extract_json('```json\n{"answer": "x"}\n```') == {"answer": "x"}


def test_json_surrounded_by_prose_is_parsed():
    text = 'Sure, here you go:\n{"answer": "x"}\nHope that helps.'
    assert extract_json(text) == {"answer": "x"}


def test_a_json_list_is_not_an_answer():
    with pytest.raises(UnparseableAnswer):
        extract_json("[1, 2, 3]")


def test_unparseable_text_raises():
    with pytest.raises(UnparseableAnswer):
        extract_json("I'm afraid I can't do that.")


# ---------- citations ----------


def test_labels_are_found_in_order_without_repeats():
    assert labels_in("a [S2] b [S1] c [S2]") == ["S2", "S1"]


def test_valid_labels_resolve_to_their_documents():
    citations, invalid = resolve_citations("rollback finished at 15:10 [S1]", SUPPLIED)
    assert invalid == ()
    assert len(citations) == 1
    assert citations[0].source_path == "ops/incidents/INC-1.pdf"


def test_an_invented_label_is_reported_not_silently_dropped():
    """A citation to a source that was never supplied is a detectable fault."""
    citations, invalid = resolve_citations("as shown [S18]", SUPPLIED)
    assert citations == ()
    assert invalid == ("S18",)


def test_labels_claimed_but_not_written_are_still_checked():
    citations, invalid = resolve_citations("no brackets here", SUPPLIED, claimed=["S2", "S99"])
    assert [c.label for c in citations] == ["S2"]
    assert invalid == ("S99",)


def test_no_citations_at_all_resolves_to_nothing():
    assert resolve_citations("plain text", SUPPLIED) == ((), ())


# ---------- the Answer object ----------


def builder() -> AnswerBuilder:
    return AnswerBuilder(question="when did the rollback finish", supplied=SUPPLIED)


def test_a_cited_answer_is_grounded():
    answer = builder().from_payload(
        {"answer": "At 15:10 [S1].", "cited_labels": ["S1"], "confidence": "high"}
    )
    assert answer.is_grounded
    assert answer.cited_documents == ("ops/incidents/INC-1.pdf",)


def test_an_uncited_answer_is_not_grounded():
    """Correct but uncheckable still fails the requirement."""
    answer = builder().from_payload({"answer": "At 15:10.", "cited_labels": []})
    assert not answer.is_grounded


def test_an_abstention_is_grounded_because_it_claims_nothing():
    answer = builder().from_payload(
        {"answer": "", "abstained": True, "abstain_reason": "no source gives a time"}
    )
    assert answer.abstained
    assert answer.is_grounded


def test_an_answer_with_an_invented_citation_is_not_grounded():
    answer = builder().from_payload({"answer": "At 15:10 [S9].", "cited_labels": ["S9"]})
    assert answer.invalid_labels == ("S9",)
    assert not answer.is_grounded


def test_contradictions_survive_into_the_answer():
    answer = builder().from_payload(
        {
            "answer": "Sources disagree [S1][S2].",
            "cited_labels": ["S1", "S2"],
            "contradictions": [
                {"description": "S1 says 15:10, S2 says 15:40", "labels": ["S1", "S2"]}
            ],
        }
    )
    assert len(answer.contradictions) == 1
    assert answer.contradictions[0].labels == ("S1", "S2")


def test_empty_contradiction_descriptions_are_discarded():
    answer = builder().from_payload(
        {"answer": "x [S1]", "contradictions": [{"description": "  ", "labels": []}]}
    )
    assert answer.contradictions == ()


def test_an_unknown_confidence_value_falls_back_to_low():
    answer = builder().from_payload({"answer": "x [S1]", "confidence": "absolutely certain"})
    assert answer.confidence == "low"


def test_a_failure_is_an_abstention_with_an_error():
    answer = builder().failure("HTTP 503")
    assert answer.failed
    assert answer.abstained
    assert not answer.is_grounded
    assert "503" in answer.error


def test_as_dict_is_serialisable():
    answer = builder().from_payload({"answer": "x [S1]", "cited_labels": ["S1"]})
    assert json.loads(json.dumps(answer.as_dict()))["citations"][0]["label"] == "S1"


# ---------- prompt rendering ----------


def test_sources_are_rendered_with_their_metadata():
    rendered = format_sources(
        [{**SUPPLIED[0], "date": "2026-03-14", "service": "payment-gateway", "kind": "incidents"}]
    )
    assert "[S1]" in rendered
    assert "date: 2026-03-14" in rendered
    assert "service: payment-gateway" in rendered


def test_no_sources_says_so_rather_than_rendering_nothing():
    assert "no sources" in format_sources([])


# ---------- the source register ----------


def test_labels_are_stable_as_sources_are_added():
    """A chunk seen as S1 must stay S1 after a tool call adds more."""
    register = SourceRegister()
    register.add([{"chunk_id": "a"}, {"chunk_id": "b"}])
    fresh = register.add([{"chunk_id": "b"}, {"chunk_id": "c"}])
    assert [e["label"] for e in register.all] == ["S1", "S2", "S3"]
    assert [e["chunk_id"] for e in fresh] == ["c"]


# ---------- the client ----------


def test_retryable_statuses_are_transient_only():
    assert {429, 500, 503} <= RETRYABLE_STATUS
    assert 400 not in RETRYABLE_STATUS
    assert 403 not in RETRYABLE_STATUS


def test_quota_exhaustion_moves_to_another_model():
    """Quotas are per model: a 429 on one says nothing about the next."""
    from ragsys.reasoning.llm import MOVE_ON_AFTER_RETRIES, MOVE_ON_IMMEDIATELY

    assert 429 in MOVE_ON_IMMEDIATELY
    assert 404 in MOVE_ON_IMMEDIATELY
    assert 503 in MOVE_ON_AFTER_RETRIES
    assert 429 not in MOVE_ON_AFTER_RETRIES


def test_an_exhausted_model_is_remembered_and_skipped():
    """Re-asking inside a spent window is a guaranteed refusal."""
    import time as _time

    from ragsys.reasoning import llm

    llm._QUOTA_BLOCKED.clear()
    assert not llm._is_quota_blocked("m")
    llm._QUOTA_BLOCKED["m"] = _time.monotonic()
    assert llm._is_quota_blocked("m")
    # Expired entries are forgotten, so a recovered model is picked back up.
    llm._QUOTA_BLOCKED["m"] = _time.monotonic() - llm.QUOTA_MEMORY_SECONDS - 1
    assert not llm._is_quota_blocked("m")
    assert "m" not in llm._QUOTA_BLOCKED


def test_an_empty_key_is_refused_before_any_request():
    with pytest.raises(LLMNotConfigured):
        GeminiClient(api_key="   ")


def test_a_text_reply_is_parsed():
    payload = {
        "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3},
    }
    parsed = GeminiClient._parse(payload)
    assert parsed.text == "hello"
    assert parsed.prompt_tokens == 11
    assert not parsed.wants_tools


def test_a_tool_call_is_parsed():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search_documents", "args": {"query": "x"}}}
                    ]
                }
            }
        ]
    }
    parsed = GeminiClient._parse(payload)
    assert parsed.wants_tools
    assert parsed.tool_calls[0].arguments == {"query": "x"}


def test_a_blocked_prompt_raises():
    with pytest.raises(LLMRefused):
        GeminiClient._parse({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})


def test_an_empty_candidate_raises():
    with pytest.raises(LLMRefused):
        GeminiClient._parse({"candidates": [{"content": {"parts": []}}]})


# ---------- the agent, against a scripted client ----------


class FakeClient:
    """Returns scripted responses, and records how it was called."""

    model = "fake"

    def __init__(self, *responses) -> None:
        self.queue = list(responses)
        self.calls: list[dict] = []

    def generate(
        self, messages, system=None, tools=None, response_schema=None, temperature=0.0
    ) -> LLMResponse:
        self.calls.append({"tools": tools, "schema": response_schema, "messages": messages})
        if not self.queue:
            raise AssertionError("the agent made more calls than were scripted")
        nxt = self.queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def reply(payload: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload), finish_reason="STOP")


class StubEmbedder:
    dim = 8

    def _vector(self, text: str) -> list[float]:
        angle = (hash(text) % 10_000) / 10_000 * 2 * math.pi
        raw = [math.cos(angle * (i + 1)) for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def retriever() -> Retriever:
    # A sparse encoder is required because the default configuration is hybrid:
    # omitting it would exercise a path the deployed system never takes.
    texts = [
        "Rollback complete at 15:10 after pool exhaustion.",
        "rollback finished about 15:40 i think",
        "name: checkout-api depends_on: payment-gateway",
    ]
    store = QdrantChunkStore(
        embedder=StubEmbedder(),
        sparse_encoder=SparseEncoder(SparseStats.from_texts(texts)),
        url=IN_MEMORY,
        collection="test_reasoning",
    )
    store.ensure_collection(recreate=True)
    chunks = []
    for i, (doc, text) in enumerate(
        [
            ("ops/incidents/INC-1.pdf", "Rollback complete at 15:10 after pool exhaustion."),
            ("ops/threads/T-1.txt", "rollback finished about 15:40 i think"),
            ("ops/services/checkout-api.json", "name: checkout-api depends_on: payment-gateway"),
        ]
    ):
        chunks.append(
            Chunk(
                chunk_id=stable_id(doc, str(i), "structural"),
                doc_id=doc,
                text=text,
                index=i,
                strategy="structural",
                metadata={"source_path": doc, "format": "md"},
            )
        )
    store.upsert(chunks)
    return Retriever(store)


def test_a_simple_question_is_answered_and_cited(retriever):
    client = FakeClient(reply({"answer": "At 15:10 [S1].", "cited_labels": ["S1"]}))
    answer = Agent(retriever, client).answer("when did the rollback finish")
    assert answer.is_grounded
    assert [c.label for c in answer.citations] == ["S1"]
    assert answer.context_size > 0


def test_the_tool_is_offered_on_the_first_call(retriever):
    client = FakeClient(reply({"answer": "x [S1]", "cited_labels": ["S1"]}))
    Agent(retriever, client).answer("anything")
    assert client.calls[0]["tools"] is not None
    assert client.calls[0]["tools"][0]["name"] == SEARCH_TOOL_NAME


def test_a_tool_call_is_run_and_the_loop_continues(retriever):
    """Multi-step: the model searches again, then answers from both rounds."""
    client = FakeClient(
        LLMResponse(text="", tool_calls=(ToolCall(SEARCH_TOOL_NAME, {"query": "checkout-api"}),)),
        reply({"answer": "checkout-api depends on it [S1]", "cited_labels": ["S1"]}),
    )
    answer = Agent(retriever, client).answer("which services depend on the payment gateway")
    assert answer.tool_calls == 1
    assert len(client.calls) == 2


def test_the_tool_budget_is_enforced(retriever):
    """A model that keeps searching is made to answer rather than looping."""
    searching = LLMResponse(text="", tool_calls=(ToolCall(SEARCH_TOOL_NAME, {"query": "more"}),))
    client = FakeClient(
        searching,
        searching,
        searching,
        reply({"answer": "done [S1]", "cited_labels": ["S1"]}),
    )
    answer = Agent(retriever, client, AgentConfig(max_tool_rounds=2)).answer("q")
    assert answer.tool_calls == 2
    assert client.calls[-1]["tools"] is None


def test_an_unknown_tool_is_reported_back_not_crashed(retriever):
    client = FakeClient(
        LLMResponse(text="", tool_calls=(ToolCall("delete_everything", {}),)),
        reply({"answer": "ok [S1]", "cited_labels": ["S1"]}),
    )
    answer = Agent(retriever, client).answer("q")
    assert not answer.failed


def test_an_empty_tool_query_is_reported_back(retriever):
    client = FakeClient(
        LLMResponse(text="", tool_calls=(ToolCall(SEARCH_TOOL_NAME, {"query": "  "}),)),
        reply({"answer": "ok [S1]", "cited_labels": ["S1"]}),
    )
    assert not Agent(retriever, client).answer("q").failed


def test_a_model_outage_becomes_a_result_not_an_exception(retriever):
    """One failure must not abort a run over the whole question set."""
    client = FakeClient(LLMUnavailable("HTTP 503 from gemini"))
    answer = Agent(retriever, client).answer("q")
    assert answer.failed
    assert answer.abstained
    assert "503" in answer.error


def test_prose_instead_of_json_falls_back_to_the_strict_schema(retriever):
    client = FakeClient(
        LLMResponse(text="I think it was around 15:10, probably."),
        reply({"answer": "At 15:10 [S1].", "cited_labels": ["S1"]}),
    )
    answer = Agent(retriever, client).answer("q")
    assert answer.is_grounded
    assert client.calls[-1]["schema"] is not None


def test_an_abstention_from_the_model_is_preserved(retriever):
    client = FakeClient(
        reply(
            {
                "answer": "",
                "abstained": True,
                "abstain_reason": "no source states the pool size",
                "cited_labels": [],
            }
        )
    )
    answer = Agent(retriever, client).answer("what is the pool minimum")
    assert answer.abstained
    assert "pool size" in answer.abstain_reason


def test_a_reported_contradiction_is_preserved(retriever):
    client = FakeClient(
        reply(
            {
                "answer": "S1 says 15:10, S2 says 15:40 [S1][S2]",
                "cited_labels": ["S1", "S2"],
                "contradictions": [
                    {"description": "the two sources disagree", "labels": ["S1", "S2"]}
                ],
            }
        )
    )
    answer = Agent(retriever, client).answer("when did the rollback finish")
    assert len(answer.contradictions) == 1


def test_an_answer_is_returned_for_every_question_shape(retriever):
    """Whatever happens, the caller gets an Answer it can record."""
    for scripted in (
        reply({"answer": "x [S1]", "cited_labels": ["S1"]}),
        reply({"answer": "", "abstained": True}),
        LLMUnavailable("down"),
        LLMRefused("safety"),
    ):
        answer = Agent(retriever, FakeClient(scripted)).answer("q")
        assert isinstance(answer, Answer)
