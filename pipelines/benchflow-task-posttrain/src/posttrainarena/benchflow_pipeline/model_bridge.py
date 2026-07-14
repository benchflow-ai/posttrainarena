"""OpenAI-compatible chat bridge for TRL's synchronized vLLM server."""

from __future__ import annotations

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

    def __post_init__(self) -> None:
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


def parse_qwen_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        body = match.group(1).strip()
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
                if not parameter_name or parameter_name in arguments:
                    raise RuntimeError(
                        f"Invalid Qwen function parameter: {parameter_name!r}"
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
        calls.append(
            {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    remaining = TOOL_CALL_PATTERN.sub("", text).strip()
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
        content = message.get("content")
        if (
            message.get("role") != "tool"
            or not isinstance(content, str)
            or len(content) <= len(TOOL_OUTPUT_TRUNCATION_MARKER)
        ):
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
        generation_kwargs.update(extra_body)
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

        async def chat_call(payload: dict[str, Any]) -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                response = await client.post(
                    f"{config.upstream_url.rstrip('/')}/chat/",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("TRL chat response is not an object")
            return data

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
        )
    )
    uvicorn.run(app, host=host, port=port)
