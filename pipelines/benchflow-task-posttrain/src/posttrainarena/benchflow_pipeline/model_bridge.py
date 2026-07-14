"""OpenAI-compatible chat bridge for TRL's synchronized vLLM server."""

from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
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


@dataclass(frozen=True)
class ModelBridgeConfig:
    upstream_url: str
    tokenizer_id: str
    tokenizer_revision: str | None = None
    api_key: str | None = None
    max_tokens_per_call: int = 4096
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


def _trl_request(body: dict[str, Any], config: ModelBridgeConfig) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    generation_kwargs = {}
    for key in ("stop", "seed", "frequency_penalty", "presence_penalty"):
        if body.get(key) is not None:
            generation_kwargs[key] = body[key]
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        generation_kwargs.update(extra_body)
    requested_max_tokens = body.get("max_completion_tokens")
    if requested_max_tokens is None:
        requested_max_tokens = body.get("max_tokens")
    max_tokens = (
        config.max_tokens_per_call
        if requested_max_tokens is None
        else min(int(requested_max_tokens), config.max_tokens_per_call)
    )
    return {
        "messages": [messages],
        "n": 1,
        "repetition_penalty": float(body.get("repetition_penalty", 1.0)),
        "temperature": float(body.get("temperature", 1.0)),
        "top_p": float(body.get("top_p", 1.0)),
        "top_k": int(body.get("top_k", -1)),
        "min_p": float(body.get("min_p", 0.0)),
        "max_tokens": max_tokens,
        "logprobs": 0,
        "generation_kwargs": generation_kwargs,
        "chat_template_kwargs": body.get("chat_template_kwargs") or {},
        "tools": body.get("tools"),
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
        upstream = await chat_call(_trl_request(body, config))
        translated = translate_trl_chat_response(
            payload=upstream,
            tokenizer=tokenizer,
            model=model,
        )
        completion_id = str(translated["id"])
        if body.get("logprobs") is True:
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
            max_sidecar_entries=max_sidecar_entries,
        )
    )
    uvicorn.run(app, host=host, port=port)
