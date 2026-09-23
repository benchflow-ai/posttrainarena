"""OpenAI-compatible chat bridge for TRL's synchronized vLLM server."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_CALL_PATTERN = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*</function>",
    re.DOTALL,
)
FUNCTION_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
TOOL_OUTPUT_TRUNCATION_MARKER = "\n...[tool output truncated]"
RESERVED_GENERATION_KWARGS = frozenset({"logprobs", "max_tokens", "n"})
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBridgeConfig:
    upstream_url: str
    tokenizer_id: str
    tokenizer_revision: str | None = None
    api_key: str | None = None
    max_tokens_per_call: int = 4096
    max_context_tokens: int = 49152
    max_logprob_context_tokens: int = 16384
    timeout_seconds: float = 900.0
    max_sidecar_entries: int = 2048
    batch_max_requests: int = 16
    batch_wait_seconds: float = 0.05

    def __post_init__(self) -> None:
        if (
            not isinstance(self.batch_max_requests, int)
            or isinstance(self.batch_max_requests, bool)
            or self.batch_max_requests < 1
        ):
            raise ValueError("batch_max_requests must be a positive integer")
        if not isinstance(self.batch_wait_seconds, int | float) or self.batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be a non-negative number")
        if (
            not isinstance(self.max_tokens_per_call, int)
            or isinstance(self.max_tokens_per_call, bool)
            or self.max_tokens_per_call < 1
        ):
            raise ValueError("max_tokens_per_call must be a positive integer")
        if (
            not isinstance(self.max_context_tokens, int)
            or isinstance(self.max_context_tokens, bool)
            or self.max_context_tokens <= self.max_tokens_per_call
        ):
            raise ValueError(
                "max_context_tokens must be an integer greater than max_tokens_per_call"
            )
        if (
            not isinstance(self.max_logprob_context_tokens, int)
            or isinstance(self.max_logprob_context_tokens, bool)
            or self.max_logprob_context_tokens <= self.max_tokens_per_call
        ):
            raise ValueError(
                "max_logprob_context_tokens must be an integer greater than "
                "max_tokens_per_call"
            )
        if (
            not isinstance(self.max_sidecar_entries, int)
            or isinstance(self.max_sidecar_entries, bool)
            or self.max_sidecar_entries < 1
        ):
            raise ValueError("max_sidecar_entries must be a positive integer")


def _parse_tool_call_body(body: str) -> dict[str, Any]:
    if body.startswith("{"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed Qwen tool call: {exc}") from exc
    else:
        function = FUNCTION_CALL_PATTERN.fullmatch(body)
        if function is None:
            raise RuntimeError("Malformed Qwen function-tag tool call")
        name = function.group(1).strip()
        arguments: dict[str, str] = {}
        parameters = function.group(2)
        consumed = 0
        for parameter in FUNCTION_PARAMETER_PATTERN.finditer(parameters):
            if parameters[consumed : parameter.start()].strip():
                raise RuntimeError("Malformed Qwen function parameter block")
            parameter_name = parameter.group(1).strip()
            if not parameter_name:
                raise RuntimeError("Invalid Qwen function parameter: ''")
            if parameter_name in arguments:
                # The policy sometimes repeats a parameter tag; keep the last value
                # instead of failing the whole completion (which aborts the rollout).
                logger.warning(
                    "Duplicate Qwen function parameter %r for %r; keeping the last value",
                    parameter_name,
                    name,
                )
            arguments[parameter_name] = parameter.group(2).strip()
            consumed = parameter.end()
        if parameters[consumed:].strip():
            raise RuntimeError("Malformed Qwen function parameter block")
        payload = {"name": name, "arguments": arguments}
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise RuntimeError("Qwen tool call has no function name")
    if not isinstance(arguments, dict):
        raise RuntimeError("Qwen tool call arguments must be an object")
    return {"name": name, "arguments": arguments}


def parse_qwen_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Split a Qwen completion into assistant text and OpenAI-shaped tool calls.

    A malformed ``<tool_call>`` block is model output, not a bridge fault: it is
    left in the text (so the agent and the training transcript see what the
    policy emitted) instead of failing the completion, which would return 500 to
    OpenCode and abort the rollout.
    """
    calls: list[dict[str, Any]] = []
    well_formed: list[tuple[int, int]] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        body = match.group(1).strip()
        try:
            payload = _parse_tool_call_body(body)
        except RuntimeError as exc:
            logger.warning("Leaving malformed Qwen tool call in the text: %s", exc)
            continue
        well_formed.append(match.span())
        calls.append(
            {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(
                        payload["arguments"],
                        separators=(",", ":"),
                    ),
                },
            }
        )
    pieces = []
    cursor = 0
    for begin, finish in well_formed:
        pieces.append(text[cursor:begin])
        cursor = finish
    pieces.append(text[cursor:])
    remaining = "".join(pieces).strip()
    return remaining or None, calls


def normalize_tool_call_arguments(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI string arguments to the mapping Qwen's template expects."""
    normalized_messages = copy.deepcopy(messages)
    for message in normalized_messages:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            function = (
                tool_call.get("function") if isinstance(tool_call, dict) else None
            )
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "OpenCode tool-call arguments are not valid JSON"
                ) from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(
                    "OpenCode tool-call arguments must decode to an object"
                )
            function["arguments"] = parsed
    return normalized_messages


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise RuntimeError("Chat template did not return token IDs")
    return value


def _prompt_token_count(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    return len(_token_ids(tokenizer.apply_chat_template(messages, **kwargs)))


def _truncatable_tool_output(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if (
        message.get("role") == "tool"
        and isinstance(content, str)
        and len(content) > len(TOOL_OUTPUT_TRUNCATION_MARKER)
    ):
        return content
    return None


def _minimal_prompt_token_count(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    minimal_messages = copy.deepcopy(messages)
    for message in minimal_messages:
        if _truncatable_tool_output(message) is not None:
            message["content"] = TOOL_OUTPUT_TRUNCATION_MARKER
    return _prompt_token_count(
        tokenizer=tokenizer,
        messages=minimal_messages,
        tools=tools,
    )


def fit_messages_to_context(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Fit a prompt by truncating oldest tool outputs, never user instructions."""
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    fitted = copy.deepcopy(messages)
    original_tokens = _prompt_token_count(
        tokenizer=tokenizer,
        messages=fitted,
        tools=tools,
    )
    if original_tokens <= max_prompt_tokens:
        return fitted, original_tokens, original_tokens, 0

    truncated_messages = 0
    for index, message in enumerate(fitted):
        content = _truncatable_tool_output(message)
        if content is None:
            continue
        message["content"] = TOOL_OUTPUT_TRUNCATION_MARKER
        truncated_messages += 1
        minimal_tokens = _prompt_token_count(
            tokenizer=tokenizer,
            messages=fitted,
            tools=tools,
        )
        if minimal_tokens > max_prompt_tokens:
            continue

        low = 0
        high = len(content)
        while low < high:
            midpoint = (low + high + 1) // 2
            fitted[index]["content"] = (
                content[:midpoint] + TOOL_OUTPUT_TRUNCATION_MARKER
            )
            tokens = _prompt_token_count(
                tokenizer=tokenizer,
                messages=fitted,
                tools=tools,
            )
            if tokens <= max_prompt_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        fitted[index]["content"] = content[:low] + TOOL_OUTPUT_TRUNCATION_MARKER
        fitted_tokens = _prompt_token_count(
            tokenizer=tokenizer,
            messages=fitted,
            tools=tools,
        )
        if fitted_tokens > max_prompt_tokens:
            raise RuntimeError(
                "OpenCode prompt context fitting did not converge "
                f"({fitted_tokens} > {max_prompt_tokens} tokens)"
            )
        return fitted, original_tokens, fitted_tokens, truncated_messages

    fitted_tokens = _prompt_token_count(
        tokenizer=tokenizer,
        messages=fitted,
        tools=tools,
    )
    if fitted_tokens > max_prompt_tokens:
        raise RuntimeError(
            "OpenCode prompt exceeds the model context after truncating all "
            f"tool outputs ({fitted_tokens} > {max_prompt_tokens} tokens)"
        )
    return fitted, original_tokens, fitted_tokens, truncated_messages


def _sampled_logprob_rows(
    *,
    completion_ids: list[int],
    logprobs: list[list[float | None]],
    logprob_token_ids: list[list[int]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    if not (len(completion_ids) == len(logprobs) == len(logprob_token_ids)):
        raise RuntimeError("TRL chat token/logprob fields are not aligned")
    rows: list[dict[str, Any]] = []
    for token_id, values, ids in zip(
        completion_ids,
        logprobs,
        logprob_token_ids,
        strict=True,
    ):
        if len(values) != len(ids):
            raise RuntimeError("TRL chat logprob candidate fields are not aligned")
        selected = next(
            (
                value
                for candidate_id, value in zip(ids, values, strict=True)
                if candidate_id == token_id
            ),
            values[0] if values else None,
        )
        if not isinstance(selected, int | float):
            raise RuntimeError(f"TRL chat has no sampled logprob for token {token_id}")
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rows.append(
            {
                "token": piece,
                "bytes": list(piece.encode("utf-8")),
                "logprob": float(selected),
                "top_logprobs": [],
                "token_id": token_id,
            }
        )
    return rows


def translate_trl_chat_response(
    *,
    payload: dict[str, Any],
    tokenizer: Any,
    model: str,
) -> dict[str, Any]:
    prompt_ids = payload.get("prompt_ids")
    completion_ids = payload.get("completion_ids")
    logprobs = payload.get("logprobs")
    logprob_token_ids = payload.get("logprob_token_ids")
    if not (
        isinstance(prompt_ids, list)
        and len(prompt_ids) == 1
        and isinstance(prompt_ids[0], list)
        and isinstance(completion_ids, list)
        and len(completion_ids) == 1
        and isinstance(completion_ids[0], list)
        and isinstance(logprobs, list)
        and len(logprobs) == 1
        and isinstance(logprobs[0], list)
        and isinstance(logprob_token_ids, list)
        and len(logprob_token_ids) == 1
        and isinstance(logprob_token_ids[0], list)
    ):
        raise RuntimeError("TRL chat returned an invalid single-completion payload")
    generated_ids = completion_ids[0]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    content, tool_calls = parse_qwen_tool_calls(text)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": {
                    "content": _sampled_logprob_rows(
                        completion_ids=generated_ids,
                        logprobs=logprobs[0],
                        logprob_token_ids=logprob_token_ids[0],
                        tokenizer=tokenizer,
                    )
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids[0]),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_ids[0]) + len(generated_ids),
        },
    }


def _trl_request(
    body: dict[str, Any],
    config: ModelBridgeConfig,
    tokenizer: Any,
) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("messages must contain objects")
    generation_kwargs = {}
    for key in ("stop", "seed", "frequency_penalty", "presence_penalty"):
        if body.get(key) is not None:
            generation_kwargs[key] = body[key]
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        generation_kwargs.update(
            {
                key: value
                for key, value in extra_body.items()
                if key not in RESERVED_GENERATION_KWARGS
            }
        )
    capture_logprobs = body.get("logprobs") is True
    if not capture_logprobs and body.get("tools") and body.get("seed") is None:
        generation_kwargs["seed"] = 0
    temperature = body.get("temperature")
    if temperature is None:
        temperature = 1.0
    requested_max_tokens = body.get("max_completion_tokens")
    if requested_max_tokens is None:
        requested_max_tokens = body.get("max_tokens")
    max_tokens = (
        config.max_tokens_per_call
        if requested_max_tokens is None
        else min(int(requested_max_tokens), config.max_tokens_per_call)
    )
    normalized_messages = normalize_tool_call_arguments(messages)
    tools = body.get("tools")
    context_tokens = config.max_context_tokens
    if capture_logprobs:
        context_tokens = min(context_tokens, config.max_logprob_context_tokens)
    minimal_prompt_tokens = _minimal_prompt_token_count(
        tokenizer=tokenizer,
        messages=normalized_messages,
        tools=tools,
    )
    if minimal_prompt_tokens >= context_tokens:
        raise RuntimeError(
            "OpenCode prompt leaves no generation capacity after truncating all "
            f"tool outputs ({minimal_prompt_tokens} >= {context_tokens} tokens)"
        )
    requested_generation_tokens = max_tokens
    max_tokens = min(max_tokens, context_tokens - minimal_prompt_tokens)
    if max_tokens < requested_generation_tokens:
        logger.warning(
            "Reduced OpenCode generation budget to fit irreducible prompt "
            "(%d -> %d max tokens; %d prompt tokens; %d context tokens)",
            requested_generation_tokens,
            max_tokens,
            minimal_prompt_tokens,
            context_tokens,
        )
    fitted_messages, original_tokens, fitted_tokens, truncated_messages = (
        fit_messages_to_context(
            tokenizer=tokenizer,
            messages=normalized_messages,
            tools=tools,
            max_prompt_tokens=context_tokens - max_tokens,
        )
    )
    if truncated_messages:
        logger.warning(
            "Truncated %d OpenCode tool output(s) to fit model context "
            "(%d -> %d prompt tokens)",
            truncated_messages,
            original_tokens,
            fitted_tokens,
        )
    return {
        "messages": [fitted_messages],
        "n": 1,
        "repetition_penalty": float(body.get("repetition_penalty", 1.0)),
        "temperature": float(temperature),
        "top_p": float(body.get("top_p", 1.0)),
        "top_k": int(body.get("top_k", -1)),
        "min_p": float(body.get("min_p", 0.0)),
        "max_tokens": max_tokens,
        "logprobs": 0,
        "generation_kwargs": generation_kwargs,
        "chat_template_kwargs": body.get("chat_template_kwargs") or {},
        "tools": tools,
    }


def _streaming_response(payload: dict[str, Any]) -> Any:
    from starlette.responses import StreamingResponse

    choice = payload["choices"][0]
    chunk = {
        **{key: payload[key] for key in ("id", "created", "model")},
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": choice["message"],
                "logprobs": choice["logprobs"],
                "finish_reason": choice["finish_reason"],
            }
        ],
    }

    async def events():
        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _is_title_request(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if body.get("tools") or not isinstance(messages, list) or len(messages) < 2:
        return False
    first = messages[0]
    second = messages[1]
    return (
        isinstance(first, dict)
        and first.get("role") == "system"
        and isinstance(first.get("content"), str)
        and first["content"].startswith("You are a title generator.")
        and isinstance(second, dict)
        and second.get("role") == "user"
        and isinstance(second.get("content"), str)
        and second["content"].startswith("Generate a title for this conversation:")
    )


def _title_response(*, model: str) -> dict[str, Any]:
    content = "Data analysis task"
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "logprobs": {"content": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


class ChatBatcher:
    """Merge concurrent single-conversation `/chat/` calls into one upstream request.

    TRL's `/chat/` endpoint does synchronous pipe I/O inside its async handler, so
    concurrent HTTP requests are served one at a time per data-parallel worker.
    Agentic rollouts (many agents, many short turns) therefore need batching on the
    client side: requests that share sampling parameters and tools are sent as one
    `messages` list, which vLLM runs with continuous batching, and the outputs are
    handed back per request. One upstream call is in flight at a time; requests that
    arrive meanwhile form the next batch.
    """

    _BATCH_KEYS = (
        "n",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "logprobs",
        "generation_kwargs",
        "chat_template_kwargs",
        "tools",
    )

    def __init__(
        self,
        post: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        *,
        max_requests: int = 16,
        wait_seconds: float = 0.05,
    ) -> None:
        self._post = post
        self._max_requests = max_requests
        self._wait_seconds = wait_seconds
        self._queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]] = (
            asyncio.Queue()
        )
        self._worker: asyncio.Task[None] | None = None
        self.upstream_calls = 0
        self.largest_batch = 0

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or len(messages) != 1 or payload.get("n", 1) != 1:
            return await self._post(payload)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        await self._queue.put((payload, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())
        return await future

    @classmethod
    def batch_key(cls, payload: dict[str, Any]) -> str:
        return json.dumps(
            {key: payload.get(key) for key in cls._BATCH_KEYS}, sort_keys=True, default=str
        )

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._queue.empty():
            batch = [await self._queue.get()]
            deadline = loop.time() + self._wait_seconds
            while len(batch) < self._max_requests:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            groups: dict[str, list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]]] = {}
            for item in batch:
                groups.setdefault(self.batch_key(item[0]), []).append(item)
            for items in groups.values():
                await self._dispatch(items)

    async def _dispatch(
        self, items: list[tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]]
    ) -> None:
        merged = dict(items[0][0])
        merged["messages"] = [payload["messages"][0] for payload, _ in items]
        self.upstream_calls += 1
        self.largest_batch = max(self.largest_batch, len(items))
        try:
            data = await self._post(merged)
            outputs = [self._slice(data, index, len(items)) for index in range(len(items))]
        except Exception as error:  # noqa: BLE001 - every waiter must be released
            for _, future in items:
                if not future.done():
                    future.set_exception(error)
            return
        for (_, future), output in zip(items, outputs, strict=True):
            if not future.done():
                future.set_result(output)

    @staticmethod
    def _slice(data: dict[str, Any], index: int, count: int) -> dict[str, Any]:
        prompt_ids = data.get("prompt_ids")
        completion_ids = data.get("completion_ids")
        if not (
            isinstance(prompt_ids, list)
            and isinstance(completion_ids, list)
            and len(prompt_ids) == count
            and len(completion_ids) == count
        ):
            raise RuntimeError(
                f"TRL chat batch returned {len(prompt_ids) if isinstance(prompt_ids, list) else '?'}"
                f" prompts and {len(completion_ids) if isinstance(completion_ids, list) else '?'}"
                f" completions for {count} conversations"
            )
        output: dict[str, Any] = {
            "prompt_ids": [prompt_ids[index]],
            "completion_ids": [completion_ids[index]],
            "logprobs": None,
            "logprob_token_ids": None,
        }
        for key in ("logprobs", "logprob_token_ids"):
            value = data.get(key)
            if isinstance(value, list):
                if len(value) != count:
                    raise RuntimeError(f"TRL chat batch returned {len(value)} {key} rows for {count} conversations")
                output[key] = [value[index]]
        return output


def create_model_bridge_app(
    config: ModelBridgeConfig,
    *,
    tokenizer: Any | None = None,
    chat_call: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
) -> Any:
    from fastapi import FastAPI, Header, HTTPException

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_id,
            revision=config.tokenizer_revision,
            trust_remote_code=True,
        )
    if chat_call is None:
        import httpx

        client = httpx.AsyncClient(timeout=config.timeout_seconds)

        async def post_chat(payload: dict[str, Any]) -> dict[str, Any]:
            response = await client.post(
                f"{config.upstream_url.rstrip('/')}/chat/",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("TRL chat response is not an object")
            return data

        batcher = ChatBatcher(
            post_chat,
            max_requests=config.batch_max_requests,
            wait_seconds=config.batch_wait_seconds,
        )
        chat_call = batcher.call

    app = FastAPI(title="PostTrain Arena TRL model bridge")
    logprob_store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def authorize(authorization: str | None) -> None:
        if config.api_key is None:
            return
        if authorization != f"Bearer {config.api_key}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/v1/models")
    async def models(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": config.tokenizer_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "posttrainarena",
                }
            ],
        }

    @app.get("/v1/benchflow/logprobs/{completion_id}")
    async def completion_logprobs(
        completion_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        payload = logprob_store.pop(completion_id, None)
        if payload is None:
            raise HTTPException(status_code=404, detail="unknown completion id")
        return payload

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> Any:
        authorize(authorization)
        model = str(body.get("model") or config.tokenizer_id)
        if _is_title_request(body):
            upstream = None
            translated = _title_response(model=model)
        else:
            upstream = await chat_call(_trl_request(body, config, tokenizer))
            translated = translate_trl_chat_response(
                payload=upstream,
                tokenizer=tokenizer,
                model=model,
            )
        completion_id = str(translated["id"])
        if body.get("logprobs") is True and upstream is not None:
            logprob_store[completion_id] = {
                "id": completion_id,
                "prompt_ids": upstream["prompt_ids"][0],
                "completion_ids": upstream["completion_ids"][0],
                "logprobs": translated["choices"][0]["logprobs"],
            }
            logprob_store.move_to_end(completion_id)
            while len(logprob_store) > config.max_sidecar_entries:
                logprob_store.popitem(last=False)
        if body.get("stream") is True:
            return _streaming_response(translated)
        return translated

    return app


def serve_model_bridge(
    *,
    upstream_url: str,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    api_key: str | None,
    max_tokens_per_call: int,
    max_context_tokens: int,
    max_logprob_context_tokens: int,
    max_sidecar_entries: int,
    host: str,
    port: int,
    batch_max_requests: int = 16,
    batch_wait_seconds: float = 0.05,
) -> None:
    import uvicorn

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url=upstream_url,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            api_key=api_key,
            max_tokens_per_call=max_tokens_per_call,
            max_context_tokens=max_context_tokens,
            max_logprob_context_tokens=max_logprob_context_tokens,
            max_sidecar_entries=max_sidecar_entries,
            batch_max_requests=batch_max_requests,
            batch_wait_seconds=batch_wait_seconds,
        )
    )
    uvicorn.run(app, host=host, port=port)
