"""Scoring answer correctness with a model that did not write the answer.

Correctness cannot be computed from a string comparison: "15:10" and "at ten
past three" are the same answer, and an answer may be right while using none
of the reference's words. So a model grades it against the reference.

The judge must not be the model under test. Language models systematically
prefer their own output, so a model grading itself measures self-consistency
rather than correctness and inflates the reported score.

Honest limitation: with one API key the judge is a different *model* from the
same provider and family, which reduces self-preference without eliminating
it. A genuinely independent judge needs a second provider, and the report says
so rather than implying otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragsys.config import settings
from ragsys.logging import get_logger
from ragsys.reasoning.agent import extract_json
from ragsys.reasoning.llm import (
    GeminiClient,
    LLMClient,
    LLMError,
    user_message,
)
from ragsys.reasoning.prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, judge_prompt

log = get_logger(__name__)

# Deliberately not the generator's default. Checked at construction.
DEFAULT_JUDGE_MODELS = ("gemini-3.5-flash", "gemini-flash-lite-latest")

VERDICTS = ("correct", "partial", "incorrect", "abstained")

# Partial credit, so one number can be reported alongside the distribution.
VERDICT_CREDIT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "abstained": 0.0}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str
    model: str

    @property
    def credit(self) -> float:
        return VERDICT_CREDIT.get(self.verdict, 0.0)


def build_judge_client(generator_model: str | None = None) -> LLMClient:
    """A client for grading, never the same model as the generator."""
    key = None
    if settings.judge_api_key is not None:
        key = settings.judge_api_key.get_secret_value()
    elif settings.llm_api_key is not None:
        key = settings.llm_api_key.get_secret_value()
        log.warning(
            "JUDGE_API_KEY is not set; grading with a different model on the "
            "generator's key. Self-preference is reduced, not eliminated."
        )

    model = settings.judge_model
    fallbacks: tuple[str, ...] = ()
    if not model:
        candidates = [m for m in DEFAULT_JUDGE_MODELS if m != generator_model]
        model, fallbacks = candidates[0], tuple(candidates[1:])
    if model == generator_model:
        log.warning(
            "the judge model (%s) is the model under test; scores will be "
            "inflated by self-preference",
            model,
        )
    return GeminiClient(model=model, api_key=key, fallbacks=fallbacks)


class Judge:
    """Grades one answer against a reference."""

    def __init__(self, client: LLMClient | None = None, generator_model: str | None = None) -> None:
        self.client = client or build_judge_client(generator_model)

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "unknown")

    def grade(
        self,
        question: str,
        reference: str | None,
        candidate: str,
        reference_is_behaviour: bool = False,
    ) -> Verdict:
        """Returns a verdict, never raises.

        A judge outage is recorded as an unscored question rather than
        aborting a run, for the same reason a generator outage is.
        """
        if not candidate.strip():
            return Verdict("abstained", "the answer was empty", self.model)
        try:
            response = self.client.generate(
                [
                    user_message(
                        judge_prompt(
                            question,
                            reference or "",
                            candidate,
                            reference_is_behaviour=reference_is_behaviour,
                        )
                    )
                ],
                system=JUDGE_SYSTEM,
                response_schema=JUDGE_SCHEMA,
                temperature=0.0,
            )
            payload = extract_json(response.text)
        except (LLMError, ValueError) as exc:
            log.warning("judge failed for %r: %s", question[:60], exc)
            return Verdict("unscored", f"judge unavailable: {exc}", self.model)

        verdict = str(payload.get("verdict", "")).lower()
        if verdict not in VERDICTS:
            verdict = "unscored"
        return Verdict(verdict, str(payload.get("reason", "")).strip(), self.model)
