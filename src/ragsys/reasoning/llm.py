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
# Groq speaks the OpenAI chat format, so this client is most of an OpenAI
# client too; only the base URL and the model names would change.
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Sent on every request. Groq sits behind Cloudflare, which rejected requests
# carrying urllib's default agent with HTTP 403 and Cloudflare error 1010 —
# a block on the client fingerprint, not on the key, and indistinguishable
# from an authentication failure without reading the body.
USER_AGENT = "ragsys/1.0 (+https://github.com/gigidz/applied-ai-investigation-system)"
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

# Reasons to stop asking this model and try the next one.
#
# Quotas are counted per model, not per key, so a 429 on one model says
# nothing about another: measured directly, gemini-3.6-flash and
# gemini-3.5-flash were exhausted while gemini-flash-lite-latest answered
# immediately. Retrying the exhausted model instead spent 66 seconds to fail,
# with a working model one line further down the list.
#
# 404 (the model does not exist for this key) and 429 (its quota is spent) are
# both immediate: no amount of waiting changes either within a request. 503 is
# a transient overload, so it is retried first and only then abandoned.
MOVE_ON_IMMEDIATELY = frozenset({404, 429})
MOVE_ON_AFTER_RETRIES = frozenset({503})

# Status codes worth trying again. 429 is rate limiting, 500/502/503/504 are
# transient server faults. A 400 or 403 is our mistake and retrying it only
# wastes the quota that is left.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_TLS_READY = False

# Requests per minute are shared across every client in the process: the
# generator and the judge draw on one quota, so pacing them independently
# would exceed it by exactly a factor of two.
_LAST_REQUEST_AT = 0.0

# Models known to be quota-limited, and when that was discovered. A quota is
# spent for a whole window, so asking again inside it is a guaranteed refusal:
# without this, every call re-tried the exhausted models first and paid about
# twelve seconds to learn what the previous call already knew.
_QUOTA_BLOCKED: dict[str, float] = {}

# Long enough to skip a per-minute window comfortably, short enough that a
# model which recovers is picked back up within a run.
QUOTA_MEMORY_SECONDS = 180.0


def _is_quota_blocked(model: str) -> bool:
    blocked_at = _QUOTA_BLOCKED.get(model)
    if blocked_at is None:
        return False
    if time.monotonic() - blocked_at > QUOTA_MEMORY_SECONDS:
        del _QUOTA_BLOCKED[model]
        return False
    return True


def _pace(requests_per_minute: int) -> None:
    """Sleep until the next request is allowed."""
    global _LAST_REQUEST_AT
    if requests_per_minute <= 0:
        return
    interval = 60.0 / requests_per_minute
    wait = interval - (time.monotonic() - _LAST_REQUEST_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_AT = time.monotonic()


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
        requests_per_minute: int | None = None,
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
        candidates = list(fallbacks) if fallbacks is not None else list(DEFAULT_GEMINI_MODELS)
        if chosen:
            # Preferred, not exclusive. Naming a model should set the order,
            # not strand the deployment on one quota: a single model with no
            # fallback is exactly the situation that produced 46 failures in
            # the first evaluation run.
            candidates = [chosen, *(m for m in candidates if m != chosen)]
        self.models = tuple(candidates)
        # The model actually used, which reporting and the evaluation report
        # need: saying "gemini-3.6-flash" when a fallback answered is a lie.
        self.model = self.models[0]
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries
        self.timeout = settings.llm_timeout_seconds if timeout is None else timeout
        self.requests_per_minute = (
            settings.llm_requests_per_minute if requests_per_minute is None else requests_per_minute
        )

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
        """Try each candidate model in turn, retrying within each.

        If every candidate is quota-limited the windows may simply need time
        to roll over, so the whole chain is attempted once more after a pause.
        A single long wait beats waiting once per model, which would multiply
        the delay by the length of the list.
        """
        _ensure_tls()
        for pass_number in (1, 2):
            failures: list[str] = []
            all_quota = True
            for model in self.models:
                if pass_number == 1 and _is_quota_blocked(model):
                    failures.append(f"{model}: quota known spent")
                    continue
                try:
                    payload = self._post_to(model, body)
                except _ModelUnusable as exc:
                    failures.append(f"{model}: {exc}")
                    if "429" in str(exc):
                        _QUOTA_BLOCKED[model] = time.monotonic()
                    else:
                        all_quota = False
                    # Logged at INFO, not WARNING. Moving to the next model is
                    # normal operation on a free tier, and the answer records
                    # which model produced it, so warning on every question
                    # would be noise in front of a working result.
                    log.info("model %s unavailable (%s); trying the next", model, exc)
                    continue
                _QUOTA_BLOCKED.pop(model, None)
                self.model = model
                return payload

            if pass_number == 1 and failures and all_quota:
                delay = settings.llm_quota_backoff_seconds
                log.warning(
                    "every model is quota-limited; waiting %.0fs for a window to roll over",
                    delay,
                )
                time.sleep(delay)
                continue
            raise LLMUnavailable("no model answered -- " + "; ".join(failures))
        raise LLMUnavailable("no model answered after waiting for quota")

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
                    "user-agent": USER_AGENT,
                },
            )
            try:
                _pace(self.requests_per_minute)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = self._describe(exc)
                if exc.code in MOVE_ON_IMMEDIATELY:
                    raise _ModelUnusable(f"HTTP {exc.code}: {detail}") from exc
                if exc.code in MOVE_ON_AFTER_RETRIES and attempt == self.max_retries:
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
            # A quota refusal needs a pause long enough for the window to
            # roll over; a transient server fault needs only a moment. Jitter
            # on both, because callers retrying in step arrive together and
            # are refused together.
            if isinstance(last, urllib.error.HTTPError) and last.code == 429:
                delay = settings.llm_quota_backoff_seconds * attempt
            else:
                delay = min(2.0**attempt, 8.0)
            time.sleep(delay * (0.5 + random.random() / 2))

        raise LLMUnavailable(f"exhausted {self.max_retries} attempts: {last}")

    @staticmethod
    def _describe(exc: urllib.error.HTTPError) -> str:
        try:
            body = json.load(exc)
        except (json.JSONDecodeError, ValueError):
            return exc.reason or "no detail"
        message = str(body.get("error", {}).get("message", ""))
        # Trimmed hard: the provider's quota message is several hundred
        # characters of billing guidance, and the status code is the
        # actionable part.
        return message.split(". ")[0][:120] or exc.reason or "no detail"

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
    if chosen == "groq":
        key = settings.judge_api_key or settings.llm_api_key
        return GroqClient(model=model, api_key=key.get_secret_value() if key else None)
    raise LLMNotConfigured(
        f"provider {chosen!r} is configured but has no client; 'gemini' and 'groq' are implemented"
    )


# Tried in order, same reasoning as the Gemini list: a free tier retires and
# rate-limits models without warning, and failing a judgement while an
# equivalent model is idle is not acceptable.
# What a free Groq key actually serves, confirmed against the models endpoint
# rather than assumed from documentation. Both are open-weight models with no
# shared lineage with the generator, which is the whole point of using a
# second provider to judge.
DEFAULT_GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
)


class GroqClient:
    """Groq's OpenAI-compatible chat API.

    Exists so the judge can be a genuinely different model family from the
    generator. Grading an answer with a sibling of the model that wrote it
    reduces self-preference without removing it, because both share training
    lineage; a Llama model grading a Gemini answer does not.

    Messages arrive in the Gemini shape, because that is what the rest of the
    package speaks, and are translated here. Keeping the translation inside
    the client means the judge and the agent need no provider-specific code.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        fallbacks: tuple[str, ...] | None = None,
        requests_per_minute: int | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMNotConfigured(
                "Groq needs an API key; set JUDGE_API_KEY in .env or pass api_key="
            )
        self._key = api_key
        candidates = list(fallbacks) if fallbacks is not None else list(DEFAULT_GROQ_MODELS)
        if model:
            candidates = [model, *(m for m in candidates if m != model)]
        self.models = tuple(candidates)
        self.model = self.models[0]
        self.max_retries = settings.llm_max_retries if max_retries is None else max_retries
        self.timeout = settings.llm_timeout_seconds if timeout is None else timeout
        self.requests_per_minute = (
            settings.llm_requests_per_minute if requests_per_minute is None else requests_per_minute
        )

    @staticmethod
    def to_openai_messages(
        messages: list[dict[str, Any]], system: str | None
    ) -> list[dict[str, str]]:
        """Translate the Gemini message shape into the OpenAI one.

        Only text parts are carried across. The judge never calls tools, and
        silently dropping a tool call would be worse than not supporting one,
        so a message carrying only tool parts becomes empty and is skipped.
        """
        converted: list[dict[str, str]] = []
        if system:
            converted.append({"role": "system", "content": system})
        for message in messages:
            text = "".join(
                str(part["text"]) for part in message.get("parts", []) if "text" in part
            ).strip()
            if not text:
                continue
            role = "assistant" if message.get("role") == "model" else "user"
            converted.append({"role": role, "content": text})
        return converted

    def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if tools:
            raise LLMNotConfigured("the Groq client is used for judging and offers no tools")

        body: dict[str, Any] = {
            "messages": self.to_openai_messages(messages, system),
            "temperature": temperature,
        }
        if response_schema is not None:
            # The OpenAI-compatible API guarantees syntactically valid JSON
            # but not the schema, so the caller still validates the fields.
            body["response_format"] = {"type": "json_object"}

        payload = self._post(body)
        return self._parse(payload)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        _ensure_tls()
        failures: list[str] = []
        for model in self.models:
            if _is_quota_blocked(model):
                failures.append(f"{model}: quota known spent")
                continue
            try:
                payload = self._post_to(model, body)
            except _ModelUnusable as exc:
                failures.append(f"{model}: {exc}")
                if "429" in str(exc):
                    _QUOTA_BLOCKED[model] = time.monotonic()
                log.info("groq model %s unavailable (%s); trying the next", model, exc)
                continue
            _QUOTA_BLOCKED.pop(model, None)
            self.model = model
            return payload
        raise LLMUnavailable("no Groq model answered -- " + "; ".join(failures))

    def _post_to(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps({**body, "model": model}).encode("utf-8")
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                GROQ_ENDPOINT,
                data=data,
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {self._key}",
                    "user-agent": USER_AGENT,
                },
            )
            try:
                _pace(self.requests_per_minute)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = self._describe(exc)
                if exc.code in MOVE_ON_IMMEDIATELY:
                    raise _ModelUnusable(f"HTTP {exc.code}: {detail}") from exc
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise LLMUnavailable(f"HTTP {exc.code} from {model}: {detail}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise LLMUnavailable(f"could not reach {model}: {exc}") from exc
                last = exc
            delay = min(2.0**attempt, 8.0)
            time.sleep(delay * (0.5 + random.random() / 2))
        raise LLMUnavailable(f"exhausted {self.max_retries} attempts: {last}")

    @staticmethod
    def _describe(exc: urllib.error.HTTPError) -> str:
        try:
            body = json.load(exc)
        except (json.JSONDecodeError, ValueError):
            return exc.reason or "no detail"
        message = str(body.get("error", {}).get("message", ""))
        return message.split(". ")[0][:120] or exc.reason or "no detail"

    @staticmethod
    def _parse(payload: dict[str, Any]) -> LLMResponse:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMRefused("Groq returned no choices")
        message = choices[0].get("message", {}) or {}
        text = str(message.get("content") or "").strip()
        if not text:
            raise LLMRefused(f"empty content (finish={choices[0].get('finish_reason')!r})")
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            finish_reason=str(choices[0].get("finish_reason", "")),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
