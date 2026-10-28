"""Tests for the evaluation harness.

A metric that is itself wrong invalidates every number in the report, so these
are worked examples with hand-computed expectations rather than snapshots of
whatever the code currently returns.

The harness is driven by fake retrieval and a fake judge, so scoring logic —
abstention credit, contradiction credit, how a model outage is recorded — is
verified without the network.
"""

from __future__ import annotations

import pytest

from ragsys.evaluation.dataset import EvalQuestion, load_hotpot_sample
from ragsys.evaluation.harness import AUTHORED, HOTPOT, Harness, group_of
from ragsys.evaluation.judge import VERDICT_CREDIT, Verdict
from ragsys.evaluation.metrics import (
    RetrievalScore,
    citation_relevance,
    citation_validity,
    found_at,
    matches,
    mean,
    mrr,
    precision_at_k,
    recall_at_k,
)
from ragsys.indexing.vector_store import Hit
from ragsys.reasoning.answer import Answer, Citation, Contradiction
from ragsys.retrieval.fusion import Candidate
from ragsys.retrieval.retriever import HYBRID_RERANKED, RetrievalResult
from ragsys.retrieval.signals import QuestionKind

# ---------- gold matching ----------


def test_an_exact_hotpot_id_matches():
    assert matches("hotpot:abc123", "hotpot:abc123")


def test_an_authored_path_suffix_matches():
    assert matches("incidents/INC-2026-03-14.pdf", "ops/incidents/INC-2026-03-14.pdf")


def test_a_different_document_does_not_match():
    assert not matches("incidents/INC-2026-03-14.pdf", "ops/incidents/INC-2026-05-11.pdf")


def test_a_prefix_is_not_a_match():
    """Suffix matching must not accept a path that merely starts the same."""
    assert not matches("ops/incidents/INC-1.pdf", "ops/incidents")


def test_empty_values_never_match():
    assert not matches("", "anything")
    assert not matches("anything", "")


# ---------- recall, mrr, precision ----------

GOLD = ["a", "b"]
RETRIEVED = ["x", "a", "y", "z", "b", "w"]


def test_found_at_is_one_based():
    assert found_at("a", RETRIEVED) == 2
    assert found_at("b", RETRIEVED) == 5
    assert found_at("missing", RETRIEVED) is None


def test_recall_counts_gold_documents_inside_the_window():
    # 'a' is at rank 2, 'b' at rank 5.
    assert recall_at_k(GOLD, RETRIEVED, 1) == 0.0
    assert recall_at_k(GOLD, RETRIEVED, 2) == 0.5
    assert recall_at_k(GOLD, RETRIEVED, 5) == 1.0


def test_recall_with_no_gold_is_zero_not_an_error():
    assert recall_at_k([], RETRIEVED, 5) == 0.0


def test_mrr_uses_the_best_rank():
    # Best gold rank is 2, so 1/2.
    assert mrr(GOLD, RETRIEVED) == pytest.approx(0.5)


def test_mrr_is_zero_when_nothing_was_found():
    assert mrr(["nope"], RETRIEVED) == 0.0


def test_precision_counts_gold_within_the_window():
    # Of the first 5 retrieved, 'a' and 'b' are gold -> 2/5.
    assert precision_at_k(GOLD, RETRIEVED, 5) == pytest.approx(0.4)


def test_precision_of_an_empty_result_is_zero():
    assert precision_at_k(GOLD, [], 5) == 0.0


def test_retrieval_score_reports_completeness():
    score = RetrievalScore.compute("q1", "simple", GOLD, RETRIEVED)
    assert score.recall_at_15 == 1.0
    assert score.rank_of_first == 2
    assert score.complete


def test_retrieval_score_is_incomplete_when_one_gold_is_missing():
    score = RetrievalScore.compute("q1", "simple", ["a", "missing"], RETRIEVED)
    assert score.recall_at_15 == 0.5
    assert not score.complete


# ---------- citation metrics ----------


def test_citation_validity_is_the_fraction_that_resolved():
    assert citation_validity(["S1", "S2", "S3"], ["S9"]) == pytest.approx(0.75)


def test_citation_validity_of_an_uncited_answer_is_zero():
    assert citation_validity([], []) == 0.0


def test_citation_relevance_measures_overlap_with_the_answer_key():
    cited = ["ops/incidents/INC-1.pdf", "ops/reports/ops-2026-03.md"]
    assert citation_relevance(cited, ["incidents/INC-1.pdf"]) == pytest.approx(0.5)


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0


# ---------- the question set ----------


def question(**kw) -> EvalQuestion:
    base = dict(
        question_id="q",
        question="?",
        answer="an answer",
        category="simple",
        gold=("incidents/INC-1.pdf",),
        source="ops/planted",
    )
    base.update(kw)
    return EvalQuestion(**base)


def test_a_contradiction_question_is_graded_on_required_behaviour():
    """The bug this guards: a null reference was read as "you should abstain".

    All four contradiction questions were marked incorrect while the separate
    contradiction measure showed the system had reported the disagreement.
    """
    q = question(
        category="contradiction",
        answer=None,
        expected_behaviour="Report both times and state that the sources disagree.",
    )
    assert q.reference == "Report both times and state that the sources disagree."
    assert q.reference_is_behaviour


def test_an_unanswerable_question_has_no_reference_to_grade_against():
    q = question(category="unanswerable", answer=None, expected_behaviour="say nothing")
    assert q.reference == ""
    assert not q.reference_is_behaviour


def test_a_normal_question_is_graded_on_its_answer():
    q = question(answer="Payments Core", expected_behaviour="look it up")
    assert q.reference == "Payments Core"
    assert not q.reference_is_behaviour


def test_the_judge_is_told_which_kind_of_reference_it_has():
    harness = Harness(FakeRetriever(["ops/x.md"]))
    judge = FakeJudge()
    harness.score_answers(
        [
            question(
                category="contradiction",
                answer=None,
                expected_behaviour="Report both and flag the conflict.",
            )
        ],
        FakeAgent(answer()),
        judge,
    )
    reference, is_behaviour = judge.seen[0]
    assert reference == "Report both and flag the conflict."
    assert is_behaviour is True


def test_only_the_unanswerable_category_expects_an_abstention():
    """A contradiction question also has a null reference, and must not count."""
    assert question(category="unanswerable", answer=None).unanswerable
    assert not question(category="contradiction", answer=None).unanswerable
    assert not question(category="simple").unanswerable


def test_contradiction_questions_are_flagged():
    assert question(category="contradiction").contradictory
    assert not question(category="simple").contradictory


def test_authored_and_sampled_questions_are_distinguished():
    assert question(source="ops/planted").is_authored
    assert not question(source="hotpotqa/distractor/validation").is_authored
    assert group_of(question(source="ops/planted")) == AUTHORED
    assert group_of(question(source="hotpotqa")) == HOTPOT


def test_a_question_survives_a_round_trip():
    original = question(category="contradiction", answer=None)
    assert EvalQuestion.from_dict(original.as_dict()) == original


def test_the_hotpot_sample_is_reproducible():
    """An evaluation whose questions change between runs proves nothing."""
    first = [q.question_id for q in load_hotpot_sample(5, seed=7)]
    second = [q.question_id for q in load_hotpot_sample(5, seed=7)]
    different = [q.question_id for q in load_hotpot_sample(5, seed=8)]
    assert first == second
    assert first != different


def test_verdict_credit_gives_partial_marks():
    assert VERDICT_CREDIT["correct"] == 1.0
    assert VERDICT_CREDIT["partial"] == 0.5
    assert VERDICT_CREDIT["incorrect"] == 0.0


# ---------- the harness ----------


class FakeRetriever:
    """Returns the documents it was told to, as retrieval candidates."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def retrieve(self, question: str, config=None) -> RetrievalResult:
        candidates = [
            Candidate(
                hit=Hit(
                    chunk_id=f"c{i}",
                    doc_id=path,
                    text="text",
                    score=1.0,
                    payload={"chunk_id": f"c{i}", "doc_id": path, "meta_source_path": path},
                )
            )
            for i, path in enumerate(self.paths)
        ]
        return RetrievalResult(
            question=question,
            kind=QuestionKind.FACTUAL,
            config_name="fake",
            candidates=candidates,
        )


class FakeJudge:
    model = "fake-judge"

    def __init__(self, verdict: str = "correct") -> None:
        self.verdict = verdict
        self.calls = 0
        self.seen: list[tuple[str, bool]] = []

    def grade(self, question, reference, candidate, reference_is_behaviour=False) -> Verdict:
        self.calls += 1
        self.seen.append((reference, reference_is_behaviour))
        return Verdict(self.verdict, "because", self.model)


class FakeAgent:
    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.client = type("C", (), {"model": "fake-generator"})()

    def answer(self, question: str) -> Answer:
        return self._answer


def test_each_rank_contributes_exactly_one_identifier():
    """Two identifiers per rank would push gold documents outside every window."""
    from ragsys.evaluation.harness import retrieved_identifiers

    retriever = FakeRetriever(["ops/incidents/INC-1.pdf", "ops/other.md"])
    candidates = retriever.retrieve("q").candidates
    assert len(retrieved_identifiers(question(), candidates)) == 2
    hotpot = question(source="hotpotqa", gold=("hotpot:x",))
    assert len(retrieved_identifiers(hotpot, candidates)) == 2


def test_a_hotpot_gold_id_is_matched_inside_the_window():
    """The bug this guards: recall read 0% while the gold was in fact retrieved."""
    harness = Harness(FakeRetriever(["hotpot:aaa", "hotpot:bbb"]))
    report = harness.score_retrieval(
        [question(source="hotpotqa", gold=("hotpot:bbb",))], HYBRID_RERANKED
    )
    stats = report.aggregate()
    assert stats["recall@5"] == 1.0
    assert stats["mrr"] == pytest.approx(0.5)


def test_retrieval_is_scored_against_the_answer_key():
    harness = Harness(FakeRetriever(["ops/incidents/INC-1.pdf", "ops/other.md"]))
    report = harness.score_retrieval([question()], HYBRID_RERANKED)
    assert report.aggregate()["recall@15"] == 1.0
    assert report.aggregate()["questions"] == 1


def test_results_are_grouped_by_source_so_they_are_never_averaged():
    harness = Harness(FakeRetriever(["ops/incidents/INC-1.pdf"]))
    report = harness.score_retrieval(
        [question(), question(source="hotpotqa", gold=("hotpot:x",))], HYBRID_RERANKED
    )
    assert set(report.groups) == {AUTHORED, HOTPOT}


def answer(**kw) -> Answer:
    base = dict(
        question="?",
        text="an answer [S1]",
        citations=(Citation("S1", "c0", "ops/incidents/INC-1.pdf", "ops/incidents/INC-1.pdf"),),
    )
    base.update(kw)
    return Answer(**base)


def test_an_unanswerable_question_is_scored_without_the_judge():
    """The right behaviour is known exactly, so a model call would be waste."""
    harness = Harness(FakeRetriever(["ops/x.md"]))
    judge = FakeJudge()
    report = harness.score_answers(
        [question(category="unanswerable", answer=None)],
        FakeAgent(answer(text="", abstained=True, citations=())),
        judge,
    )
    assert judge.calls == 0
    assert report.outcomes[0].verdict == "correct"


def test_answering_an_unanswerable_question_is_incorrect():
    harness = Harness(FakeRetriever(["ops/x.md"]))
    report = harness.score_answers(
        [question(category="unanswerable", answer=None)],
        FakeAgent(answer()),
        FakeJudge(),
    )
    assert report.outcomes[0].verdict == "incorrect"


def test_abstention_precision_and_recall_are_distinguished():
    harness = Harness(FakeRetriever(["ops/x.md"]))
    report = harness.score_answers(
        [
            question(question_id="u1", category="unanswerable", answer=None),
            question(question_id="s1"),
        ],
        FakeAgent(answer(text="", abstained=True, citations=())),
        FakeJudge(),
    )
    stats = report.aggregate()
    # It abstained on both: right once, wrong once.
    assert stats["abstention_recall"] == 1.0
    assert stats["abstention_precision"] == pytest.approx(0.5)


def test_contradiction_detection_is_measured_only_where_expected():
    harness = Harness(FakeRetriever(["ops/x.md"]))
    reported = answer(contradictions=(Contradiction("they disagree", ("S1", "S2")),))
    report = harness.score_answers(
        [question(category="contradiction", answer=None)], FakeAgent(reported), FakeJudge()
    )
    assert report.aggregate()["contradiction_detection"] == 1.0


def test_a_missed_contradiction_scores_zero():
    harness = Harness(FakeRetriever(["ops/x.md"]))
    report = harness.score_answers(
        [question(category="contradiction", answer=None)], FakeAgent(answer()), FakeJudge()
    )
    assert report.aggregate()["contradiction_detection"] == 0.0


def test_a_model_outage_is_recorded_as_unscored_not_incorrect():
    """A failed call is missing data, not a wrong answer."""
    harness = Harness(FakeRetriever(["ops/x.md"]))
    broken = answer(text="", citations=(), abstained=True, error="HTTP 503")
    report = harness.score_answers([question()], FakeAgent(broken), FakeJudge())
    assert report.outcomes[0].verdict == "unscored"
    assert report.aggregate()["model_failures"] == 1


def test_a_run_resumes_from_saved_progress(tmp_path):
    """The binding constraint is a daily quota, so lost work is expensive."""
    progress = tmp_path / "progress.jsonl"
    questions = [question(question_id="q1"), question(question_id="q2")]
    harness = Harness(FakeRetriever(["ops/incidents/INC-1.pdf"]))

    first = harness.score_answers(
        questions[:1], FakeAgent(answer()), FakeJudge(), progress=progress
    )
    assert len(first.outcomes) == 1

    judge = FakeJudge()
    second = harness.score_answers(questions, FakeAgent(answer()), judge, progress=progress)
    assert len(second.outcomes) == 2
    # q1 was replayed from disk, so only q2 reached the judge.
    assert judge.calls == 1


def test_a_failed_question_is_retried_rather_than_replayed(tmp_path):
    progress = tmp_path / "progress.jsonl"
    questions = [question(question_id="q1")]
    harness = Harness(FakeRetriever(["ops/incidents/INC-1.pdf"]))
    broken = answer(text="", citations=(), abstained=True, error="HTTP 429")

    harness.score_answers(questions, FakeAgent(broken), FakeJudge(), progress=progress)
    judge = FakeJudge()
    again = harness.score_answers(questions, FakeAgent(answer()), judge, progress=progress)
    assert judge.calls == 1
    assert not again.outcomes[0].failed


def test_citation_metrics_reach_the_report():
    harness = Harness(FakeRetriever(["ops/x.md"]))
    report = harness.score_answers([question()], FakeAgent(answer()), FakeJudge())
    stats = report.aggregate()
    assert stats["citation_validity"] == 1.0
    assert stats["citation_relevance"] == 1.0
