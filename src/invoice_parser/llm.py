"""The only place that talks to a model.

Three providers sit behind one interface, chosen by name:

    create_llm("vllm")     an OpenAI-compatible endpoint you host (vLLM on Modal or any GPU box)
    create_llm("openai")   the OpenAI API
    create_llm("claude")   the Claude API

Everything else in the package depends only on `LLM`: ask for JSON matching a schema, or for
plain text, optionally with page images. Retries, saved answers, usage accounting and the
offline switch are shared; a provider only says how to send one request.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("invoice_parser")

HEARTBEAT_SECONDS = 30

ENV_PROVIDER = "INVOICE_PARSER_PROVIDER"
ENV_URL = "INVOICE_PARSER_LLM_URL"
ENV_KEY = "INVOICE_PARSER_LLM_KEY"
ENV_MODEL = "INVOICE_PARSER_MODEL"
ENV_EFFORT = "INVOICE_PARSER_EFFORT"

DEFAULT_PROVIDER = "vllm"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # the open-weight model served by modal_app/serve.py
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"

# Hosted reasoning models spend output tokens on thinking before they answer, so the room a
# caller asks for is doubled and never set below this.
HOSTED_MIN_OUTPUT_TOKENS = 16000


class LLMError(RuntimeError):
    pass


class NotConfigured(LLMError):
    """No model has been set up at all (as opposed to one set up wrongly)."""


class Truncated(LLMError):
    """The model ran out of output tokens; the caller should send a smaller request."""


class LLM(Protocol):
    def json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        tag: str,
        max_tokens: int = 4096,
        images: list[bytes] | None = None,
    ) -> T: ...

    def text(
        self,
        *,
        system: str,
        user: str,
        tag: str,
        max_tokens: int = 4096,
        images: list[bytes] | None = None,
    ) -> str: ...

    def usage(self) -> dict[str, Any]: ...


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# --- shared machinery -------------------------------------------------------------------------


class Usage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_tag: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, tag: str, **values: float) -> None:
        with self._lock:
            for k, v in values.items():
                self._by_tag[tag][k] += v

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            by_tag = {t: {k: round(v, 2) for k, v in d.items()} for t, d in self._by_tag.items()}
        total: dict[str, float] = defaultdict(float)
        for d in by_tag.values():
            for k, v in d.items():
                total[k] += v
        return {"total": {k: round(v, 2) for k, v in total.items()}, "by_tag": by_tag}

    def reset(self) -> None:
        with self._lock:
            self._by_tag.clear()


class DirCache:
    """Saved model answers as one small JSON file each, in a folder."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> str | None:
        file = self.path / f"{key}.json"
        return file.read_text() if file.is_file() else None

    def set(self, key: str, value: str) -> None:
        file = self.path / f"{key}.json"
        tmp = file.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(value)
        tmp.replace(file)


class KeyValueCache:
    """Saved model answers in any store with get(key) and set(key, value), such as a Redis
    client, so that workers on different machines share them."""

    def __init__(self, client: Any, prefix: str = "invoice-parser:llm:") -> None:
        self.client = client
        self.prefix = prefix

    def get(self, key: str) -> str | None:
        value = self.client.get(self.prefix + key)
        return value.decode() if isinstance(value, bytes | bytearray) else value

    def set(self, key: str, value: str) -> None:
        self.client.set(self.prefix + key, value)


def make_cache(cache: Any) -> DirCache | KeyValueCache | None:
    """A folder path, a get/set store, or nothing."""
    if cache is None or isinstance(cache, DirCache | KeyValueCache):
        return cache
    if isinstance(cache, str | Path):
        return DirCache(Path(cache))
    if hasattr(cache, "get") and hasattr(cache, "set"):
        return KeyValueCache(cache)
    raise TypeError("cache must be a folder path or an object with get(key) and set(key, value)")


def strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """A model's JSON Schema in the strict form hosted APIs require for guaranteed output:
    every object closed to extra keys, every property listed as required. A field that may be
    absent is already nullable, so "required" only means the model must write null."""
    out = copy.deepcopy(schema.model_json_schema())

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            node.pop("default", None)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(out)
    return out


@dataclass
class Reply:
    """What one request came back with, whatever the provider."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False


class BaseLLM:
    """Shared behaviour: saved answers, usage, retries, the offline switch, waiting notices.

    A provider subclasses this and implements `_send`. It may also override `_retryable` to say
    which of its errors are worth another attempt.
    """

    provider = "base"

    def __init__(self, model: str, cache: Any = None, retries: int = 4) -> None:
        self.model = model
        self.retries = retries
        self.offline = False  # when set, a question not already answered on disk is an error
        self._cache = make_cache(cache)
        self._usage = Usage()
        self._lock = threading.Lock()
        self._in_flight: dict[int, float] = {}
        self._next_id = 0
        self._answered = False
        self._beating = False  # the waiting-notice thread runs only while requests are out

    # --- what a provider supplies ---------------------------------------------------------

    def _send(
        self,
        system: str,
        user: str,
        images: list[bytes] | None,
        max_tokens: int,
        schema: type[BaseModel] | None,
    ) -> Reply:
        raise NotImplementedError

    def _retryable(self, exc: Exception) -> bool:
        """Whether a failed request should be tried again here. Providers whose own client
        already retries return False."""
        return False

    def _cache_fields(self, schema: type[BaseModel] | None) -> dict[str, Any]:
        """What, besides the prompt, makes two requests the same question."""
        return {
            "provider": self.provider,
            "schema": schema.model_json_schema() if schema else None,
        }

    # --- the interface --------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        return {**self._usage.snapshot(), "provider": self.provider, "model": self.model}

    def reset_usage(self) -> None:
        self._usage.reset()

    def json(self, *, system, user, schema, tag, max_tokens=4096, images=None):
        raw = self._complete(system, user, tag, max_tokens, images, schema)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(f"{tag}: model output did not match {schema.__name__}: {exc}") from exc

    def text(self, *, system, user, tag, max_tokens=4096, images=None):
        return self._complete(system, user, tag, max_tokens, images, None)

    # --- internals ------------------------------------------------------------------------

    def _heartbeat(self) -> None:
        """Say so when requests have been waiting a while (cold starts are silent otherwise).
        Ends by itself once nothing is waiting, so an idle client holds no thread."""
        while True:
            time.sleep(HEARTBEAT_SECONDS)
            with self._lock:
                waits = [time.time() - t for t in self._in_flight.values()]
                if not waits:
                    self._beating = False
                    return
            if waits and max(waits) >= HEARTBEAT_SECONDS:
                hint = "" if self._answered else " (no answer yet: the model may be starting up)"
                log.info(
                    "waiting on model: %d call(s) in flight, longest %.0fs%s",
                    len(waits),
                    max(waits),
                    hint,
                )

    def _cache_key(self, payload: dict[str, Any], images: list[bytes] | None) -> str | None:
        if self._cache is None:
            return None
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        for img in images or []:
            h.update(hashlib.sha256(img).digest())
        return h.hexdigest()[:32]

    def _complete(
        self,
        system: str,
        user: str,
        tag: str,
        max_tokens: int,
        images: list[bytes] | None,
        schema: type[BaseModel] | None,
    ) -> str:
        key = {
            "model": self.model, "system": system, "user": user, "max_tokens": max_tokens,
            **self._cache_fields(schema),
        }  # fmt: skip
        cache_key = self._cache_key(key, images)
        saved = self._cache.get(cache_key) if cache_key else None
        if saved is not None:
            self._usage.add(tag, calls=1, cached=1)
            return json.loads(saved)["content"]
        if self.offline:
            raise LLMError(
                f"{tag}: not in the saved answers, and offline mode forbids a model call"
            )

        with self._lock:
            self._next_id += 1
            call_id = self._next_id
        for attempt in range(self.retries + 1):
            started = time.time()
            with self._lock:
                self._in_flight[call_id] = started
                if not self._beating:
                    self._beating = True
                    threading.Thread(target=self._heartbeat, daemon=True).start()
            try:
                reply = self._send(system, user, images, max_tokens, schema)
            except LLMError:
                raise
            except Exception as exc:
                if not self._retryable(exc) or attempt == self.retries:
                    raise LLMError(
                        f"{self.provider} request failed: {type(exc).__name__}: {exc}"
                    ) from exc
                delay = min(60, 3 * 2**attempt)
                log.info(
                    "model call %s failed after %.0fs (%s: %s); retry %d/%d in %ds",
                    tag, time.time() - started, type(exc).__name__, str(exc)[:160],
                    attempt + 1, self.retries, delay,
                )  # fmt: skip
                time.sleep(delay)
                continue
            finally:
                with self._lock:
                    self._in_flight.pop(call_id, None)
            self._answered = True
            self._usage.add(
                tag,
                calls=1,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                seconds=time.time() - started,
            )
            log.debug(
                "model call %s: %.1fs, %s prompt + %s output tokens%s",
                tag, time.time() - started, reply.prompt_tokens, reply.completion_tokens,
                ", truncated" if reply.truncated else "",
            )  # fmt: skip
            if reply.truncated:
                raise Truncated(f"{tag}: output exceeded the token limit")
            if cache_key:
                self._cache.set(
                    cache_key, json.dumps({"tag": tag, "content": reply.text}, ensure_ascii=False)
                )
            return reply.text
        raise LLMError(
            f"{self.provider} request failed"
        )  # unreachable; keeps the type checker calm


# --- providers --------------------------------------------------------------------------------


def _image_urls(images: list[bytes] | None) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()},
        }
        for img in images or []
    ]


class VLLMClient(BaseLLM):
    """A model you host behind an OpenAI-compatible endpoint: vLLM, or anything that speaks the
    same chat API and can hold its output to a JSON schema."""

    provider = "vllm"

    def __init__(
        self,
        base_url: str,
        api_key: str = "none",
        model: str = DEFAULT_MODEL,
        cache: Any = None,
        timeout: float = 600.0,
        retries: int = 4,
    ) -> None:
        from openai import OpenAI

        # Accept the bare deployment URL as well as one that already ends in /v1.
        base_url = base_url.strip().rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        super().__init__(model, cache, retries)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    @staticmethod
    def _format(schema: type[BaseModel] | None) -> dict[str, Any] | None:
        if schema is None:
            return None
        return {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }

    def _cache_fields(self, schema):
        # The shape answers were first saved under; kept so existing saved answers stay valid.
        return {"format": self._format(schema)}

    def _retryable(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)  # none for network errors and cold starts
        return status is None or status >= 500 or status in (408, 425, 429)

    def _send(self, system, user, images, max_tokens, schema):
        content: Any = [{"type": "text", "text": user}, *_image_urls(images)] if images else user
        extra = {"response_format": self._format(schema)} if schema else {}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=max_tokens,
            **extra,
        )
        choice, used = resp.choices[0], resp.usage
        return Reply(
            text=choice.message.content or "",
            prompt_tokens=getattr(used, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(used, "completion_tokens", 0) or 0,
            truncated=choice.finish_reason == "length",
        )


class OpenAIClient(BaseLLM):
    """The OpenAI API, using its strict structured-output mode."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: Any = None,
        timeout: float = 600.0,
        retries: int = 4,
    ) -> None:
        from openai import OpenAI

        super().__init__(model, cache, retries=0)  # the OpenAI client retries by itself
        # With no key given the client reads OPENAI_API_KEY from the environment.
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries
        )

    def _send(self, system, user, images, max_tokens, schema):
        content: Any = [{"type": "text", "text": user}, *_image_urls(images)] if images else user
        extra: dict[str, Any] = {}
        if schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": strict_schema(schema),
                    "strict": True,
                },
            }
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_completion_tokens=max(2 * max_tokens, HOSTED_MIN_OUTPUT_TOKENS),
            **extra,
        )
        choice, used = resp.choices[0], resp.usage
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"the model declined the request: {choice.message.refusal}")
        return Reply(
            text=choice.message.content or "",
            prompt_tokens=getattr(used, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(used, "completion_tokens", 0) or 0,
            truncated=choice.finish_reason == "length",
        )


class ClaudeClient(BaseLLM):
    """The Claude API, using structured outputs.

    Credentials come from the Anthropic SDK's usual sources (ANTHROPIC_API_KEY, or a profile
    from `ant auth login`) unless a key is passed.
    """

    provider = "claude"
    # Models on which a request declined by a safety classifier can be re-run server-side.
    _FALLBACK_MODELS = frozenset(
        {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
    )
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        api_key: str | None = None,
        cache: Any = None,
        effort: str | None = "low",
        fallbacks: bool = True,
        timeout: float = 600.0,
        retries: int = 4,
        base_url: str | None = None,
    ) -> None:
        import anthropic

        super().__init__(model, cache, retries=0)  # the Anthropic client retries by itself
        options: dict[str, Any] = {"timeout": timeout, "max_retries": retries}
        if api_key:
            options["api_key"] = api_key
        if base_url and base_url != "http://offline.invalid":
            options["base_url"] = base_url
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(**options)
        self.effort = effort
        self.fallbacks = fallbacks and model in self._FALLBACK_MODELS

    def _cache_fields(self, schema):
        return {**super()._cache_fields(schema), "effort": self.effort}

    def _send(self, system, user, images, max_tokens, schema):
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(img).decode(),
                },
            }
            for img in images or []
        ]
        content.append({"type": "text", "text": user})
        request: dict[str, Any] = {
            "model": self.model,
            # Thinking shares this budget with the answer.
            "max_tokens": max(2 * max_tokens, HOSTED_MIN_OUTPUT_TOKENS),
            # The instructions are the same for every part of a document: cache them.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
        }
        output_config: dict[str, Any] = {"effort": self.effort} if self.effort else {}
        if schema is not None:
            # The schema is sent as is and the reply validated by the caller, not by the SDK's
            # parse helper: a reply cut off at the token limit must surface as "truncated" so
            # the caller can retry with a smaller request, not as a parsing error.
            output_config["format"] = {
                "type": "json_schema",
                "schema": self._anthropic.transform_schema(schema),
            }
        if output_config:
            request["output_config"] = output_config
        messages = self._client.messages
        if self.fallbacks:
            messages = self._client.beta.messages
            request["betas"] = [self._FALLBACK_BETA]
            request["fallbacks"] = "default"
        anthropic = self._anthropic
        try:
            response = messages.create(**request)
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Claude rejected the request: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude credentials are missing or invalid") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"Claude credentials lack permission: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Claude does not know the model {self.model!r}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached, and retries were used up") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Claude API: {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            reason = getattr(details, "explanation", None) or getattr(details, "category", None)
            raise LLMError(f"the model declined the request: {reason or 'no reason given'}")
        used = response.usage
        prompt_tokens = (
            (used.input_tokens or 0)
            + (getattr(used, "cache_read_input_tokens", 0) or 0)
            + (getattr(used, "cache_creation_input_tokens", 0) or 0)
        )
        truncated = response.stop_reason == "max_tokens"
        text = "".join(b.text for b in response.content if b.type == "text")
        return Reply(text, prompt_tokens, used.output_tokens or 0, truncated)


# --- choosing a provider ------------------------------------------------------------------------

Factory = Callable[..., BaseLLM]


def _make_vllm(*, model=None, base_url=None, api_key=None, cache=None, **options) -> BaseLLM:
    base_url = base_url or os.environ.get(ENV_URL)
    if not base_url:
        raise NotConfigured(
            f"{ENV_URL} is not set. Deploy modal_app/serve.py (or start any OpenAI-compatible "
            f"server) and put its URL and {ENV_KEY} in the environment or .env."
        )
    return VLLMClient(
        base_url,
        api_key or os.environ.get(ENV_KEY, "none"),
        model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL,
        cache,
        **options,
    )


def _make_openai(*, model=None, base_url=None, api_key=None, cache=None, **options) -> BaseLLM:
    model = model or os.environ.get(ENV_MODEL)
    if not model:
        raise LLMError(f"the openai provider needs a model name: pass model=... or set {ENV_MODEL}")
    return OpenAIClient(model, api_key, base_url, cache, **options)


def _make_claude(*, model=None, base_url=None, api_key=None, cache=None, **options) -> BaseLLM:
    options.setdefault("effort", os.environ.get(ENV_EFFORT) or "low")
    model = model or os.environ.get(ENV_MODEL) or DEFAULT_CLAUDE_MODEL
    return ClaudeClient(model, api_key, cache, base_url=base_url, **options)


PROVIDERS: dict[str, Factory] = {
    "vllm": _make_vllm,
    "openai": _make_openai,
    "claude": _make_claude,
}


def register_provider(name: str, factory: Factory) -> None:
    """Add a provider of your own. `factory(model=, base_url=, api_key=, cache=, **options)` returns an
    object with the `LLM` interface; subclassing `BaseLLM` gives you the shared behaviour."""
    PROVIDERS[name] = factory


def create_llm(
    provider: str | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    cache: Any = None,
    offline: bool = False,
    **options: Any,
) -> BaseLLM:
    """The model client for a provider.

    Every setting can be passed here, which is how to supply values fetched at run time (from
    a secrets manager, say); what is passed always wins. Anything left out comes from the
    environment: INVOICE_PARSER_PROVIDER (default vllm), INVOICE_PARSER_MODEL, and the
    provider's own credentials."""
    load_dotenv()
    name = (provider or os.environ.get(ENV_PROVIDER) or DEFAULT_PROVIDER).lower()
    if name not in PROVIDERS:
        raise LLMError(f"unknown provider {name!r}; choose from {', '.join(sorted(PROVIDERS))}")
    if offline:
        # Only saved answers will be used, so no real endpoint or credential is needed.
        base_url, api_key = base_url or "http://offline.invalid", api_key or "offline"
        if name == "openai":
            model = model or os.environ.get(ENV_MODEL) or "offline"
    llm = PROVIDERS[name](
        model=model, base_url=base_url, api_key=api_key, cache=make_cache(cache), **options
    )
    llm.offline = offline
    return llm


class FakeLLM:
    """Test double: one handler per tag, receiving (system, user) and returning a dict or model."""

    def __init__(self, handlers: dict[str, Callable[[str, str], Any]] | None = None) -> None:
        self.handlers = handlers or {}
        self.calls: list[tuple[str, str]] = []
        self._usage = Usage()

    def usage(self) -> dict[str, Any]:
        return self._usage.snapshot()

    def reset_usage(self) -> None:
        self._usage.reset()

    def _run(self, tag: str, system: str, user: str) -> Any:
        self.calls.append((tag, user))
        self._usage.add(tag, calls=1)
        if tag not in self.handlers:
            raise LLMError(f"FakeLLM has no handler for {tag!r}")
        return self.handlers[tag](system, user)

    def json(self, *, system, user, schema, tag, max_tokens=4096, images=None):
        out = self._run(tag, system, user)
        return out if isinstance(out, schema) else schema.model_validate(out)

    def text(self, *, system, user, tag, max_tokens=4096, images=None):
        return str(self._run(tag, system, user))
