"""Running the question set and aggregating what happened.

Two passes, kept separate because their costs differ by three orders of
magnitude. Scoring retrieval is arithmetic over lists and runs the whole set
against four configurations in seconds. Scoring answers costs one generation
call and one judge call per question, which is the only part constrained by a
free tier's daily quota.

Authored and HotpotQA results are aggregated separately and never averaged.
They measure different properties over incomparable data: 7,405 externally
labelled Wikipedia questions against 27 questions over documents we wrote. A
blended number would let a good retrieval score conceal bad contradiction
handling.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragsys.evaluation.dataset import EvalQuestion
from ragsys.evaluation.judge import Judge, Verdict
from ragsys.evaluation.metrics import (
    RetrievalScore,
    citation_relevance,
    citation_validity,
    mean,
)
from ragsys.logging import get_logger
from ragsys.reasoning.agent import Agent
from ragsys.reasoning.answer import Answer
from ragsys.retrieval.retriever import RetrievalConfig, Retriever

log = get_logger(__name__)

AUTHORED = "authored"
HOTPOT = "hotpot"


def group_of(question: EvalQuestion) -> str:
    return AUTHORED if question.is_authored else HOTPOT


# ---------- retrieval ----------


def retrieved_identifiers(question: EvalQuestion, candidates: Sequence[Any]) -> list[str]:
    """Express the retrieved list in the same currency as the answer key.

    The two sources identify documents differently: an authored question names
    a path suffix such as `incidents/INC-2026-03-14.pdf`, while a HotpotQA
    question names a derived paragraph id such as `hotpot:197f8b11e019e335`.

    One identifier per rank, which is the whole point. An earlier version
    concatenated both forms into one list, so fifteen results became thirty
    entries and every HotpotQA gold document landed at rank 16 or later —
    outside every @k window. Recall read 0.0% while the rank-unbounded
    measures still found the documents, and the inconsistency is what exposed
    it. Rank-based metrics require exactly one entry per retrieved item.
    """
    if question.is_authored:
        return [c.hit.source_path for c in candidates]
    return [c.doc_id for c in candidates]


@dataclass
class RetrievalReport:
    config_name: str
    scores: list[RetrievalScore] = field(default_factory=list)
    seconds: float = 0.0
    groups: dict[str, list[RetrievalScore]] = field(default_factory=dict)

    def aggregate(self, subset: Sequence[RetrievalScore] | None = None) -> dict[str, Any]:
        rows = list(subset if subset is not None else self.scores)
        if not rows:
            return {}
        return {
            "questions": len(rows),
            "recall@5": mean([r.recall_at_5 for r in rows]),
            "recall@10": mean([r.recall_at_10 for r in rows]),
            "recall@15": mean([r.recall_at_15 for r in rows]),
            "mrr": mean([r.mrr for r in rows]),
            "precision@15": mean([r.precision_at_15 for r in rows]),
            "all_gold_found": mean([1.0 if r.complete else 0.0 for r in rows]),
            "distinct_documents": mean([float(r.distinct_documents) for r in rows]),
        }

    def by_category(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[RetrievalScore]] = {}
        for score in self.scores:
            buckets.setdefault(score.category, []).append(score)
        return {name: self.aggregate(rows) for name, rows in sorted(buckets.items())}

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_name,
            "seconds": round(self.seconds, 2),
            "overall": self.aggregate(),
            "by_group": {k: self.aggregate(v) for k, v in sorted(self.groups.items())},
            "by_category": self.by_category(),
        }


# ---------- answers ----------


@dataclass(frozen=True)
class AnswerOutcome:
    question_id: str
    category: str
    group: str
    verdict: str
    reason: str
    abstained: bool
    should_abstain: bool
    contradiction_expected: bool
    contradiction_reported: bool
    grounded: bool
    citation_validity: float
    citation_relevance: float
    citations: int
    invalid_citations: int
    tool_calls: int
    tokens: int
    failed: bool
    error: str

    @property
    def credit(self) -> float:
        return {"correct": 1.0, "partial": 0.5}.get(self.verdict, 0.0)


@dataclass
class AnswerReport:
    config_name: str
    generator_model: str
    judge_model: str
    outcomes: list[AnswerOutcome] = field(default_factory=list)
    seconds: float = 0.0

    def aggregate(self, subset: Sequence[AnswerOutcome] | None = None) -> dict[str, Any]:
        rows = list(subset if subset is not None else self.outcomes)
        if not rows:
            return {}

        verdicts: dict[str, int] = {}
        for row in rows:
            verdicts[row.verdict] = verdicts.get(row.verdict, 0) + 1

        should = [r for r in rows if r.should_abstain]
        did = [r for r in rows if r.abstained]
        expected_contradictions = [r for r in rows if r.contradiction_expected]
        cited = [r for r in rows if r.citations or r.invalid_citations]

        return {
            "questions": len(rows),
            "verdicts": dict(sorted(verdicts.items())),
            "correctness": mean([r.credit for r in rows]),
            "grounded": mean([1.0 if r.grounded else 0.0 for r in rows]),
            # Of the citations written, how many pointed at a supplied source.
            "citation_validity": mean([r.citation_validity for r in cited]) if cited else 0.0,
            "citation_relevance": mean([r.citation_relevance for r in cited]) if cited else 0.0,
            # Did it abstain where it should have?
            "abstention_recall": mean([1.0 if r.abstained else 0.0 for r in should])
            if should
            else None,
            # Of the times it abstained, how often was that right?
            "abstention_precision": mean([1.0 if r.should_abstain else 0.0 for r in did])
            if did
            else None,
            "contradiction_detection": mean(
                [1.0 if r.contradiction_reported else 0.0 for r in expected_contradictions]
            )
            if expected_contradictions
            else None,
            "model_failures": sum(1 for r in rows if r.failed),
            "mean_tokens": mean([float(r.tokens) for r in rows]),
            "mean_tool_calls": mean([float(r.tool_calls) for r in rows]),
        }

    def by_category(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[AnswerOutcome]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.category, []).append(outcome)
        return {name: self.aggregate(rows) for name, rows in sorted(buckets.items())}

    def as_dict(self) -> dict[str, Any]:
        groups: dict[str, list[AnswerOutcome]] = {}
        for outcome in self.outcomes:
            groups.setdefault(outcome.group, []).append(outcome)
        return {
            "config": self.config_name,
            "generator_model": self.generator_model,
            "judge_model": self.judge_model,
            "seconds": round(self.seconds, 2),
            "overall": self.aggregate(),
            "by_group": {k: self.aggregate(v) for k, v in sorted(groups.items())},
            "by_category": self.by_category(),
            "failures": [
                {"question_id": o.question_id, "error": o.error} for o in self.outcomes if o.failed
            ],
            # Per-question detail, because the brief asks for an analysis of
            # where the system fails and why, and an aggregate cannot say
            # which question went wrong or what the judge objected to.
            "questions": [asdict(o) for o in self.outcomes],
        }


class Harness:
    """Scores retrieval, and optionally answers, over a question set."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def score_retrieval(
        self, questions: Sequence[EvalQuestion], config: RetrievalConfig
    ) -> RetrievalReport:
        started = time.perf_counter()
        report = RetrievalReport(config_name=config.name)
        for question in questions:
            result = self.retriever.retrieve(question.question, config)
            retrieved = retrieved_identifiers(question, result.candidates)
            score = RetrievalScore.compute(
                question_id=question.question_id,
                category=question.category,
                gold=question.gold,
                retrieved=retrieved,
            )
            report.scores.append(score)
            report.groups.setdefault(group_of(question), []).append(score)
        report.seconds = time.perf_counter() - started
        return report

    def score_answers(
        self,
        questions: Sequence[EvalQuestion],
        agent: Agent,
        judge: Judge,
        config_name: str = "hybrid+rerank",
        progress: Path | None = None,
    ) -> AnswerReport:
        """Score every question, optionally resuming a previous run.

        Resumable because the binding constraint is a daily quota, not time.
        An unpaced first attempt exhausted the free tier after one question
        and lost the other 46; appending each outcome as it completes means a
        run interrupted at question 30 continues from 31 tomorrow rather than
        starting again and spending the new quota on work already done.
        """
        started = time.perf_counter()
        report = AnswerReport(
            config_name=config_name,
            generator_model=getattr(agent.client, "model", "unknown"),
            judge_model=judge.model,
        )

        done: dict[str, AnswerOutcome] = {}
        if progress and progress.exists():
            for line in progress.open(encoding="utf-8"):
                outcome = AnswerOutcome(**json.loads(line))
                done[outcome.question_id] = outcome
            # A failed question is not a finished question: retry those.
            done = {k: v for k, v in done.items() if not v.failed}
            log.info("resuming with %d questions already scored", len(done))

        handle = progress.open("a", encoding="utf-8") if progress else None
        try:
            for number, question in enumerate(questions, start=1):
                if question.question_id in done:
                    report.outcomes.append(done[question.question_id])
                    continue
                answer = agent.answer(question.question)
                verdict = self._verdict_for(question, answer, judge)
                outcome = self._outcome(question, answer, verdict)
                report.outcomes.append(outcome)
                if handle:
                    handle.write(json.dumps(asdict(outcome)) + "\n")
                    handle.flush()
                log.info(
                    "[%d/%d] %s -> %s",
                    number,
                    len(questions),
                    question.question_id,
                    verdict.verdict,
                )
        finally:
            if handle:
                handle.close()

        report.seconds = time.perf_counter() - started
        return report

    # ---------- internals ----------

    @staticmethod
    def _verdict_for(question: EvalQuestion, answer: Answer, judge: Judge) -> Verdict:
        """Unanswerable questions are scored in code, not by the judge.

        The correct behaviour is known exactly — abstain — so asking a model
        is both a wasted call and a less reliable answer than a boolean.
        """
        if question.unanswerable:
            if answer.abstained:
                return Verdict("correct", "correctly declined to answer", judge.model)
            return Verdict(
                "incorrect",
                "answered a question with no answer in the corpus",
                judge.model,
            )
        if answer.failed:
            return Verdict("unscored", answer.error, judge.model)
        return judge.grade(
            question.question,
            question.reference,
            answer.text,
            reference_is_behaviour=question.reference_is_behaviour,
        )

    @staticmethod
    def _outcome(question: EvalQuestion, answer: Answer, verdict: Verdict) -> AnswerOutcome:
        return AnswerOutcome(
            question_id=question.question_id,
            category=question.category,
            group=group_of(question),
            verdict=verdict.verdict,
            reason=verdict.reason,
            abstained=answer.abstained,
            should_abstain=question.unanswerable,
            contradiction_expected=question.contradictory,
            contradiction_reported=bool(answer.contradictions),
            grounded=answer.is_grounded,
            citation_validity=citation_validity(
                [c.label for c in answer.citations], answer.invalid_labels
            ),
            citation_relevance=citation_relevance(answer.cited_documents, question.gold),
            citations=len(answer.citations),
            invalid_citations=len(answer.invalid_labels),
            tool_calls=answer.tool_calls,
            tokens=answer.prompt_tokens + answer.output_tokens,
            failed=answer.failed,
            error=answer.error,
        )
