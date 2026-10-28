"""The loop that turns a question into a grounded answer.

Retrieval runs once before the model is involved, which answers most
questions. The model may then call `search_documents` for anything the first
retrieval missed, and that is where multi-step reasoning actually happens: for
"which change caused the March disruption and which services were affected",
the second query is only knowable after reading the answer to the first.

Structured output and tool calling cannot be combined in one request against
this API, so the output contract is carried in the prompt and parsed here,
with the provider's strict schema kept as a fallback for a reply that will not
parse. That ordering is deliberate — asking for the schema first would mean
giving up tool calling, and tool calling is a requirement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ragsys.logging import get_logger
from ragsys.reasoning.answer import Answer, AnswerBuilder
from ragsys.reasoning.llm import (
    LLMClient,
    LLMError,
    LLMResponse,
    build_client,
    model_message,
    tool_result_message,
    user_message,
)
from ragsys.reasoning.prompts import (
    RESPONSE_SCHEMA,
    SYSTEM,
    extra_sources_prompt,
    question_prompt,
)
from ragsys.reasoning.tools import SEARCH_TOOL_NAME, SourceRegister, ToolBox
from ragsys.retrieval.retriever import Retriever

log = get_logger(__name__)

# A fenced code block around JSON, which models emit despite being asked not to.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class UnparseableAnswer(ValueError):
    """The reply was not JSON, after every repair this module attempts."""


@dataclass(frozen=True)
class AgentConfig:
    """How much freedom the model gets, and how much it may cost."""

    # Two rounds is enough for the two-hop questions in the evaluation set and
    # bounds the worst case: a model that keeps searching would otherwise spend
    # the free-tier quota on one question.
    max_tool_rounds: int = 2
    tool_results: int = 6
    temperature: float = 0.0


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply.

    Three attempts, cheapest first: the whole string, the contents of a
    markdown fence, then the outermost braces. Models are inconsistent about
    wrapping, and a retry costs a network round trip while this costs nothing.
    """
    candidates = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise UnparseableAnswer(text[:300])


class Agent:
    """Retrieval, optional self-directed search, then one grounded answer."""

    def __init__(
        self,
        retriever: Retriever,
        client: LLMClient | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.client = client or build_client()
        self.config = config or AgentConfig()

    def answer(self, question: str) -> Answer:
        register = SourceRegister()
        toolbox = ToolBox(self.retriever, register, limit=self.config.tool_results)

        first = self.retriever.retrieve(question)
        register.add(first.as_context())

        messages = [user_message(question_prompt(question, register.all))]
        builder = AnswerBuilder(question=question, supplied=register.all)
        rounds = 0

        try:
            response = self._ask(messages, tools=toolbox.declarations)
            while response.wants_tools and rounds < self.config.max_tool_rounds:
                rounds += 1
                messages.append(model_message(response.text or "(calling a tool)"))
                for call in response.tool_calls:
                    messages.append(self._run_tool(call, toolbox))
                builder.supplied = register.all
                # On the last permitted round there is no point offering a
                # tool that cannot be run: the model would ask, be refused,
                # and cost a round trip. Measured at one wasted call in four
                # on a question needing two searches.
                remaining = rounds < self.config.max_tool_rounds
                response = self._ask(messages, tools=toolbox.declarations if remaining else None)

            if response.wants_tools:
                # Out of rounds and still asking. Answer from what is in hand
                # rather than looping, and say so in the transcript.
                log.info("tool budget exhausted for %r; answering from current sources", question)
                messages.append(user_message("Answer now, using the sources already supplied."))
                response = self._ask(messages, tools=None)

            builder.supplied = register.all
            builder.tool_calls = rounds
            builder.prompt_tokens = response.prompt_tokens
            builder.output_tokens = response.output_tokens
            # Read after the call, not before: the client advances through its
            # fallback list, so the model that answered is not necessarily the
            # one that was asked first.
            builder.model = getattr(self.client, "model", "")

            try:
                payload = extract_json(response.text)
            except UnparseableAnswer:
                log.warning("reply was not JSON; retrying with a strict schema")
                payload = self._ask_with_schema(messages)

            return builder.from_payload(payload)

        except LLMError as exc:
            # A model outage is recorded as a result for this question, not
            # raised, so one failure cannot abort a run over the whole set.
            log.warning("model failed for %r: %s", question, exc)
            builder.supplied = register.all
            builder.model = getattr(self.client, "model", "")
            return builder.failure(str(exc))

    # ---------- internals ----------

    def _ask(self, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        return self.client.generate(
            messages,
            system=SYSTEM,
            tools=tools,
            temperature=self.config.temperature,
        )

    def _ask_with_schema(self, messages: list[dict]) -> dict:
        """Last resort: the provider's own JSON enforcement, without tools."""
        response = self.client.generate(
            messages + [user_message("Reply with the JSON object only.")],
            system=SYSTEM,
            response_schema=RESPONSE_SCHEMA,
            temperature=self.config.temperature,
        )
        try:
            return extract_json(response.text)
        except UnparseableAnswer as exc:
            raise LLMError(f"model never produced parseable JSON: {exc}") from exc

    def _run_tool(self, call, toolbox: ToolBox) -> dict:
        if call.name != SEARCH_TOOL_NAME:
            return tool_result_message(call.name, f"no such tool: {call.name}")
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return tool_result_message(call.name, "the query argument was empty")
        fresh = toolbox.search(query)
        log.info("tool search %r returned %d new sources", query, len(fresh))
        if not fresh:
            return tool_result_message(
                call.name, f"no sources beyond those already supplied matched {query!r}"
            )
        return tool_result_message(call.name, extra_sources_prompt(query, fresh))
