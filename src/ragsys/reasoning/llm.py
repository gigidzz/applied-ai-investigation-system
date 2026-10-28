"""Talking to a language model over HTTP, with failures treated as normal.

Written against the REST API with the standard library rather than a vendor
SDK, for three reasons: the deployed image stays small, there is no second
dependency to pin against a free-tier host, and tool calling is a handful of
JSON fields rather than a framework.

Every failure mode here was observed rather than imagined. A free tier returns
429 under load, Google returns 503 when a model is busy, and this interpreter
cannot complete a TLS handshake at all without `truststore`, because the
python.org build ships no certificate bundle on macOS.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from ragsys.config import settings
from ragsys.logging import get_logger

log = get_logger(__name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Tried in order. Two failure modes made this a list rather than a string,
# both observed against this key within a minute of each other: a retired
# alias answers 404 naming its replacement, and a popular model answers 503
# "experiencing high demand" however long you back off. Neither is worth
# failing a question over while an equivalent model is idle.
DEFAULT_GEMINI_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
)

# A model that does not exist for this key is no more usable than one that is
# overloaded, so both move on to the next candidate.
UNUSABLE_MODEL_STATUS = frozenset({404, 503})

# Status codes worth trying again. 429 is rate limiting, 500/502/503/504 are
# transient server faults. A 400 or 403 is our mistake and retrying it only
# wastes the quota that is left.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_TLS_READY = False


def _ensure_tls() -> None:
    """Install the system certificate store, once, before the first request.

    Done lazily rather than at import so that importing this module stays free
    of side effects and the test suite never touches the network stack.
    """
    global _TLS_READY
    if _TLS_READY:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:  # pragma: no cover - present in this project's lock file
        log.warning("truststore is unavailable; TLS verification may fail")
    _TLS_READY = True


class LLMError(RuntimeError):
    """Base class, so callers can handle every model failure in one place."""


class LLMNotConfigured(LLMError):
    """No API key available."""


class LLMUnavailable(LLMError):
    """The model could not be reached, after exhausting retries."""


class LLMRefused(LLMError):
    """The model returned no usable content, for example on a safety stop."""


class LLMBadResponse(LLMError):
    """The response arrived but could not be parsed."""


class _ModelUnusable(LLMError):
    """Internal: this model cannot serve the request, but another might."""


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run one of the tools it was offered."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """The only model interface the reasoning stage depends on.

    Narrow on purpose: one method, plain data in and out. That is what lets
    the tests drive the whole agent with a scripted fake and no network.
    """

    model: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class GeminiClient:
    """Google Generative Language API over plain HTTP."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        fallbacks: tuple[str, ...] | None = None,
    ) -> None:
        key = api_key
        if key is None and settings.llm_api_key is not None:
            key = settings.llm_api_key.get_secret_value()
        if not key or not key.strip():
            raise LLMNotConfigured(
                "Set LLM_API_KEY in .env to a Google AI Studio key, or pass api_key="
            )
        self._key = key
        chosen = model or settings.llm_model
        if chosen:
            self.models = (chosen, *(fallbacks or ()))
        else:
            self.models = tuple(fallbacks) if fallbacks else DEFAULT_GEMINI_MODELS
        # The model actually used, which reporting and the evaluation report
        # need: saying "gemini-3.6-flash" when a fallback answered is a lie.
        self.model = self.models[0]
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries
        self.timeout = settings.llm_timeout_seconds if timeout is None else timeout

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "contents": messages,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
        if response_schema is not None:
            # Structured output and tool calling are mutually exclusive here:
            # a model asked for strict JSON cannot also emit a function call,
            # so the agent asks for a schema only on its final turn.
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = response_schema

        payload = self._post(body)
        return self._parse(payload)

    # ---------- transport ----------

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Try each candidate model in turn, retrying within each."""
        _ensure_tls()
        failures: list[str] = []
        for model in self.models:
            try:
                payload = self._post_to(model, body)
            except _ModelUnusable as exc:
                failures.append(f"{model}: {exc}")
                log.warning("model %s is unusable (%s); trying the next", model, exc)
                continue
            self.model = model
            return payload
        raise LLMUnavailable("no model answered -- " + "; ".join(failures))

    def _post_to(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = GEMINI_ENDPOINT.format(model=model)
        data = json.dumps(body).encode("utf-8")
        last: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "content-type": "application/json",
                    "x-goog-api-key": self._key,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = self._describe(exc)
                if exc.code in UNUSABLE_MODEL_STATUS and (
                    exc.code == 404 or attempt == self.max_retries
                ):
                    raise _ModelUnusable(f"HTTP {exc.code}: {detail}") from exc
                if exc.code not in RETRYABLE_STATUS:
                    raise LLMUnavailable(f"HTTP {exc.code} from {model}: {detail}") from exc
                if attempt == self.max_retries:
                    raise LLMUnavailable(f"HTTP {exc.code} from {model}: {detail}") from exc
                last = exc
                log.warning(
                    "attempt %d/%d failed with HTTP %d (%s); retrying",
                    attempt,
                    self.max_retries,
                    exc.code,
                    detail,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise LLMUnavailable(f"could not reach {model}: {exc}") from exc
                last = exc
                log.warning(
                    "attempt %d/%d could not reach the model (%s); retrying",
                    attempt,
                    self.max_retries,
                    exc,
                )
            # Exponential backoff with jitter: several callers retrying in step
            # would arrive together and be rate limited together.
            time.sleep(min(2.0**attempt, 8.0) * (0.5 + random.random() / 2))

        raise LLMUnavailable(f"exhausted {self.max_retries} attempts: {last}")

    @staticmethod
    def _describe(exc: urllib.error.HTTPError) -> str:
        try:
            body = json.load(exc)
        except (json.JSONDecodeError, ValueError):
            return exc.reason or "no detail"
        return str(body.get("error", {}).get("message", ""))[:200] or "no detail"

    # ---------- response shape ----------

    @staticmethod
    def _parse(payload: dict[str, Any]) -> LLMResponse:
        candidates = payload.get("candidates") or []
        if not candidates:
            blocked = payload.get("promptFeedback", {}).get("blockReason")
            raise LLMRefused(f"no candidates returned (blockReason={blocked!r})")

        candidate = candidates[0]
        finish = str(candidate.get("finishReason", ""))
        parts = candidate.get("content", {}).get("parts") or []

        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                texts.append(str(part["text"]))
            call = part.get("functionCall")
            if call:
                calls.append(
                    ToolCall(name=str(call.get("name", "")), arguments=dict(call.get("args") or {}))
                )

        if not texts and not calls:
            raise LLMRefused(f"response carried no text and no tool call (finish={finish!r})")

        usage = payload.get("usageMetadata") or {}
        return LLMResponse(
            text="".join(texts).strip(),
            tool_calls=tuple(calls),
            finish_reason=finish,
            prompt_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
        )


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def model_message(text: str) -> dict[str, Any]:
    return {"role": "model", "parts": [{"text": text}]}


def tool_result_message(name: str, result: str) -> dict[str, Any]:
    """A tool's output, in the shape the API expects to see it returned."""
    return {
        "role": "user",
        "parts": [{"functionResponse": {"name": name, "response": {"result": result}}}],
    }


def build_client(provider: str | None = None, model: str | None = None) -> LLMClient:  # noqa: D401
    """One place that knows which providers exist."""
    chosen = provider or settings.llm_provider
    if chosen == "gemini":
        return GeminiClient(model=model)
    raise LLMNotConfigured(
        f"provider {chosen!r} is configured but only 'gemini' has a client; "
        "add one here or set LLM_PROVIDER=gemini"
    )
