"""The instructions and the output contract.

The prompt is the only place the system's behavioural requirements are
actually enforced, so each rule here maps to something the evaluation harness
measures: citation accuracy, abstention on insufficient evidence, and
reporting disagreement instead of silently resolving it.

Output is a fixed JSON schema rather than prose for the same reason. Parsing
"I'm not sure, but..." to decide whether the system abstained is guesswork;
reading a boolean is not.
"""

from __future__ import annotations

from typing import Any

SYSTEM = """\
You are an investigation assistant answering questions about a document \
collection. You will be given numbered sources. Follow these rules exactly.

GROUNDING
- Use only the supplied sources. Never use your own knowledge of the world, \
even if you are confident it is correct.
- Put a citation immediately after each claim, in square brackets, using the \
source label: "the pool was reduced to 10 [S3]".
- A sentence drawing on two sources carries both: "... [S1][S4]".
- Never cite a label you were not given.

INSUFFICIENT EVIDENCE
- If the sources do not contain the answer, set "abstained" to true, leave \
"answer" empty or state plainly that the documents do not say, and explain \
what is missing in "abstain_reason".
- Partial evidence is not sufficient evidence. Answer the part that is \
supported, and say explicitly which part is not.
- Do not guess, infer beyond the text, or fill gaps with plausible detail.

DISAGREEMENT
- If two sources state different facts about the same thing, do not choose \
between them and do not average them.
- Report both values, attribute each to its source, and add an entry to \
"contradictions" naming the sources that disagree.

REASONING
- For a question needing several pieces of information, use the \
search_documents tool to look for what is missing before answering.
- Record what you did in "steps", one short line per step.

CONFIDENCE
- "high" when the sources state the answer directly.
- "medium" when it requires combining sources.
- "low" when the evidence is thin, indirect, or contradictory.

OUTPUT
Reply with one JSON object and nothing else. No prose before or after, no
markdown fences.

{"answer": "text with [Sn] citations",
 "cited_labels": ["S1"],
 "abstained": false,
 "abstain_reason": "",
 "contradictions": [{"description": "S1 says 15:10, S4 says 15:40", "labels": ["S1", "S4"]}],
 "steps": ["searched for the release", "read the post-mortem timeline"],
 "confidence": "high"}

If you call search_documents instead, call it and wait for the result rather
than also replying with JSON.\
"""

# Gemini's structured-output schema, an OpenAPI subset with upper-case types.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "answer": {
            "type": "STRING",
            "description": "The answer, with a [Sn] citation after each claim.",
        },
        "cited_labels": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Every source label used, for example ['S1', 'S4'].",
        },
        "abstained": {
            "type": "BOOLEAN",
            "description": "True if the sources do not support an answer.",
        },
        "abstain_reason": {
            "type": "STRING",
            "description": "What is missing, when abstaining. Empty otherwise.",
        },
        "contradictions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "description": {"type": "STRING"},
                    "labels": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["description", "labels"],
            },
        },
        "steps": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "answer",
        "cited_labels",
        "abstained",
        "abstain_reason",
        "contradictions",
        "steps",
        "confidence",
    ],
}


def format_sources(context: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as a labelled source list.

    The metadata is included because it is what the model needs to attribute a
    statement to a document and to notice that two sources describing the same
    event carry different dates.
    """
    if not context:
        return "(no sources were retrieved)"
    blocks = []
    for entry in context:
        header = f"[{entry['label']}] {entry.get('source_path', entry.get('doc_id', ''))}"
        facts = []
        if entry.get("date"):
            facts.append(f"date: {entry['date']}")
        if entry.get("service"):
            facts.append(f"service: {entry['service']}")
        if entry.get("kind"):
            facts.append(f"type: {entry['kind']}")
        if facts:
            header += f"  ({', '.join(facts)})"
        text = " ".join(str(entry.get("text", "")).split())
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def question_prompt(question: str, context: list[dict[str, Any]]) -> str:
    return (
        f"SOURCES\n{'=' * 70}\n{format_sources(context)}\n{'=' * 70}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the sources above, citing each claim."
    )


def extra_sources_prompt(query: str, context: list[dict[str, Any]]) -> str:
    """Appended when the model's own tool call brings back more evidence."""
    return (
        f"ADDITIONAL SOURCES for your search {query!r}\n"
        f"{'=' * 70}\n{format_sources(context)}\n{'=' * 70}"
    )


JUDGE_SYSTEM = """\
You are grading one answer against a reference answer. Reply with JSON only.

Score "correct" if the answer states the same facts as the reference, even in \
different words. Score "partial" if it states some of them and contradicts \
none. Score "incorrect" if it contradicts the reference or misses its \
substance. Score "abstained" if the answer declines to answer.

Extra detail beyond the reference is not an error. Different wording is not an \
error. A missing or wrong central fact is.\
"""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {
            "type": "STRING",
            "enum": ["correct", "partial", "incorrect", "abstained"],
        },
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "reason"],
}


def judge_prompt(
    question: str, reference: str, candidate: str, reference_is_behaviour: bool = False
) -> str:
    """Assemble the grading request.

    `reference_is_behaviour` matters for contradiction questions, where no
    single answer is correct and the key describes what the answer must *do*
    instead. Grading "report both figures and note the discrepancy" as though
    it were a factual answer marks a correct response wrong.
    """
    if reference_is_behaviour:
        header = (
            "REQUIRED BEHAVIOUR (not a single factual answer -- score "
            "'correct' if the answer does what this describes)"
        )
    else:
        header = "REFERENCE ANSWER"
    return (
        f"QUESTION: {question}\n\n"
        f"{header}: {reference or '(the correct behaviour is to abstain)'}\n\n"
        f"ANSWER TO GRADE: {candidate or '(empty)'}"
    )
