"""The question set the system is scored against.

Two sources, combined into one file so the count is visible in one place
rather than needing explanation. The authored operations questions cover the
behaviours HotpotQA cannot test at all — synthesis over related documents,
planted contradictions, and answers that are deliberately absent — while
HotpotQA supplies externally labelled multi-step questions in a volume that
makes retrieval numbers mean something.

The HotpotQA sample is drawn with a fixed seed. An evaluation whose question
set changes between runs cannot support a claim that a change improved
anything.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ragsys.config import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
OPS_FILE = EVAL_DIR / "ops_questions.jsonl"
HOTPOT_FILE = EVAL_DIR / "hotpot_questions.jsonl"
COMBINED_FILE = EVAL_DIR / "test_set.jsonl"

# Same seed as the corpus build, so the whole data pipeline is reproducible
# from one number.
SEED = 20260314

# Enough HotpotQA questions to carry the statistical weight while keeping a
# full answer-scoring run inside a free tier's daily quota: every question
# costs one generation call plus one judge call.
DEFAULT_HOTPOT_SAMPLE = 20


@dataclass(frozen=True)
class EvalQuestion:
    """One scored question, from either source."""

    question_id: str
    question: str
    answer: str | None
    category: str
    gold: tuple[str, ...]
    source: str
    expected_behaviour: str = ""

    @property
    def unanswerable(self) -> bool:
        """The correct behaviour is to abstain.

        Keyed on the category alone, deliberately. A contradiction question
        also carries a null reference answer — because no single value is
        correct — but abstaining on one is a failure, not a success: the
        required behaviour is to report both values and attribute each. An
        earlier version tested `answer is None` and so counted all four
        contradiction questions as abstentions, which would have credited the
        system for refusing to answer them.
        """
        return self.category == "unanswerable"

    @property
    def contradictory(self) -> bool:
        """The correct behaviour is to report disagreement, not pick a side."""
        return self.category == "contradiction"

    @property
    def is_authored(self) -> bool:
        return self.source.startswith("ops/")

    @property
    def reference(self) -> str:
        """What the judge should grade the answer against.

        A contradiction question has a null reference answer, because no
        single value is correct — and that null must not be read as "there is
        nothing to say". The answer key already describes the required
        behaviour ("report both times, attribute each to its source"), so
        that description is the reference.

        The first run of this harness got this wrong: a null reference was
        rendered to the judge as "the correct behaviour is to abstain", which
        is true of an unanswerable question and false of a contradictory one.
        All four contradiction questions were marked incorrect while the
        separate contradiction-detection measure showed the system had in
        fact reported the disagreement on half of them.
        """
        if self.answer:
            return str(self.answer)
        if self.contradictory and self.expected_behaviour:
            return self.expected_behaviour
        return ""

    @property
    def reference_is_behaviour(self) -> bool:
        """True when the reference describes required behaviour, not a fact."""
        return not self.answer and bool(self.contradictory and self.expected_behaviour)

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "gold": list(self.gold),
            "source": self.source,
            "expected_behaviour": self.expected_behaviour,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> EvalQuestion:
        return cls(
            question_id=str(raw["question_id"]),
            question=str(raw["question"]),
            answer=raw.get("answer"),
            category=str(raw.get("category", "unknown")),
            gold=tuple(str(g) for g in (raw.get("gold") or [])),
            source=str(raw.get("source", "")),
            expected_behaviour=str(raw.get("expected_behaviour", "")),
        )


def load_ops(path: Path = OPS_FILE) -> list[EvalQuestion]:
    """The authored questions, whose gold keys are path suffixes."""
    questions = []
    for line in path.open(encoding="utf-8"):
        raw = json.loads(line)
        questions.append(
            EvalQuestion(
                question_id=str(raw["question_id"]),
                question=str(raw["question"]),
                answer=raw.get("answer"),
                category=str(raw["category"]),
                gold=tuple(str(g) for g in (raw.get("gold_documents") or [])),
                source=str(raw.get("source", "ops/planted")),
                expected_behaviour=str(raw.get("expected_behaviour", "")),
            )
        )
    return questions


def load_hotpot_sample(
    count: int = DEFAULT_HOTPOT_SAMPLE,
    seed: int = SEED,
    path: Path = HOTPOT_FILE,
) -> list[EvalQuestion]:
    """A reproducible sample, whose gold keys are derived paragraph ids."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    chosen = random.Random(seed).sample(rows, min(count, len(rows)))
    # Sorted after sampling so the file order is stable regardless of how the
    # sampler happened to walk the list.
    chosen.sort(key=lambda r: r["question_id"])
    return [
        EvalQuestion(
            question_id=str(raw["question_id"]),
            question=str(raw["question"]),
            answer=raw.get("answer"),
            category=str(raw.get("category", "multi_hop")),
            gold=tuple(str(g) for g in (raw.get("gold_doc_ids") or [])),
            source=str(raw.get("source", "hotpotqa")),
            expected_behaviour=str(raw.get("reasoning_type", "")),
        )
        for raw in chosen
    ]


def build(hotpot_count: int = DEFAULT_HOTPOT_SAMPLE, seed: int = SEED) -> list[EvalQuestion]:
    return load_ops() + load_hotpot_sample(hotpot_count, seed)


def save(questions: list[EvalQuestion], path: Path = COMBINED_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.as_dict(), ensure_ascii=False) + "\n")


def load(path: Path = COMBINED_FILE) -> list[EvalQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_test_set.py to create it")
    return [EvalQuestion.from_dict(json.loads(line)) for line in path.open(encoding="utf-8")]


def summarise(questions: list[EvalQuestion]) -> dict[str, int]:
    return dict(sorted(Counter(q.category for q in questions).items()))
