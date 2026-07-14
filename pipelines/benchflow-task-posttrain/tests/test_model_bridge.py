from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from posttrainarena.benchflow_pipeline.model_bridge import (
    ModelBridgeConfig,
    create_model_bridge_app,
    normalize_tool_call_arguments,
    parse_qwen_tool_calls,
    translate_trl_chat_response,
)


class FakeTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return bytes(token_ids).decode("utf-8")


def _upstream(text: str) -> dict[str, Any]:
    token_ids = list(text.encode("utf-8"))
    return {
        "prompt_ids": [[1, 2, 3]],
        "completion_ids": [token_ids],
        "logprobs": [[[-0.25] for _ in token_ids]],
        "logprob_token_ids": [[[token_id] for token_id in token_ids]],
    }


def test_parse_qwen_tool_calls_returns_openai_shape() -> None:
    content, calls = parse_qwen_tool_calls(
        'thinking\n<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
    )

    assert content == "thinking"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pwd"}


def test_parse_qwen35_function_tag_tool_calls() -> None:
    content, calls = parse_qwen_tool_calls(
        """
thinking
<tool_call>
<function=bash>
<parameter=command>
sqlite3 database.sqlite ".tables"
</parameter>
<parameter=timeout>
30
</parameter>
</function>
</tool_call>
"""
    )

    assert content == "thinking"
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "command": 'sqlite3 database.sqlite ".tables"',
        "timeout": "30",
    }


def test_parse_qwen35_rejects_malformed_function_parameters() -> None:
    with pytest.raises(RuntimeError, match="parameter block"):
        parse_qwen_tool_calls(
            """
<tool_call>
<function=bash>
unexpected
<parameter=command>pwd</parameter>
</function>
</tool_call>
"""
        )


def test_normalize_tool_call_arguments_for_qwen_chat_template() -> None:
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"filePath":"/home/user/input/data.csv"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "a,b\n1,2",
        },
    ]

    normalized = normalize_tool_call_arguments(messages)

    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {
        "filePath": "/home/user/input/data.csv"
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"filePath":"/home/user/input/data.csv"}'
    )


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_normalize_tool_call_arguments_rejects_invalid_values(arguments: str) -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": arguments,
                    },
                }
            ],
        }
    ]

    with pytest.raises(RuntimeError, match="arguments"):
        normalize_tool_call_arguments(messages)


def test_translate_trl_chat_response_preserves_token_ids_and_logprobs() -> None:
    payload = translate_trl_chat_response(
        payload=_upstream("OK"),
        tokenizer=FakeTokenizer(),
        model="Qwen/Qwen3-4B",
    )

    choice = payload["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "OK"}
    assert choice["finish_reason"] == "stop"
    assert [row["token_id"] for row in choice["logprobs"]["content"]] == [
        ord("O"),
        ord("K"),
    ]
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_translate_qwen35_function_tag_response_emits_tool_call() -> None:
    payload = translate_trl_chat_response(
        payload=_upstream(
            """
<tool_call>
<function=bash>
<parameter=command>pwd</parameter>
</function>
</tool_call>
"""
        ),
        tokenizer=FakeTokenizer(),
        model="Qwen/Qwen3.5-9B",
    )

    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "command": "pwd"
    }


def test_model_bridge_serves_authenticated_streaming_tool_call() -> None:
    captured: dict[str, Any] = {}

    async def fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return _upstream(
            '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
        )

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
            api_key="secret",
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fake_chat,
    )
    client = TestClient(app)
    request = {
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "inspect"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "stream": True,
        "max_tokens": 32,
        "logprobs": True,
    }

    assert client.post("/v1/chat/completions", json=request).status_code == 401
    response = client.post(
        "/v1/chat/completions",
        json=request,
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    chunk = json.loads(events[0])
    tool_call = chunk["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash"
    assert chunk["choices"][0]["finish_reason"] == "tool_calls"
    assert chunk["choices"][0]["logprobs"]["content"]
    assert events[-1] == "[DONE]"
    completion_id = chunk["id"]
    sidecar = client.get(
        f"/v1/benchflow/logprobs/{completion_id}",
        headers={"Authorization": "Bearer secret"},
    )
    assert sidecar.status_code == 200
    assert sidecar.json()["prompt_ids"] == [1, 2, 3]
    assert sidecar.json()["completion_ids"][:2] == [ord("<"), ord("t")]
    assert sidecar.json()["logprobs"]["content"]
    assert (
        client.get(
            f"/v1/benchflow/logprobs/{completion_id}",
            headers={"Authorization": "Bearer secret"},
        ).status_code
        == 404
    )
    assert captured["payload"]["messages"] == [request["messages"]]
    assert captured["payload"]["tools"] == request["tools"]
    assert captured["payload"]["logprobs"] == 0
    assert captured["payload"]["max_tokens"] == 32
    assert captured["payload"]["temperature"] == 1.0
    assert "seed" not in captured["payload"]["generation_kwargs"]


def test_model_bridge_short_circuits_title_helper_without_provider_call() -> None:
    async def fail_chat(_payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("title helper must not call the model")

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fail_chat,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a title generator. You output ONLY a thread title."
                    ),
                },
                {
                    "role": "user",
                    "content": "Generate a title for this conversation:\n",
                },
                {"role": "user", "content": "Analyze red wine quality."},
            ],
            "stream": True,
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    chunk = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert chunk["choices"][0]["delta"]["content"] == "Data analysis task"
    assert chunk["choices"][0]["finish_reason"] == "stop"
    assert (
        TestClient(app)
        .get(
            f"/v1/benchflow/logprobs/{chunk['id']}",
        )
        .status_code
        == 404
    )


def test_model_bridge_does_not_intercept_tool_bearing_title_prompt() -> None:
    captured: dict[str, Any] = {}

    async def fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return _upstream("OK")

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fake_chat,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a title generator.",
                },
                {
                    "role": "user",
                    "content": "Generate a title for this conversation:",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {"type": "object"}},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured["payload"]["tools"]


def test_model_bridge_normalizes_followup_tool_arguments_for_trl() -> None:
    captured: dict[str, Any] = {}

    async def fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return _upstream("OK")

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fake_chat,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"filePath":"/tmp/data.csv"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "a,b\n1,2",
                },
            ],
        },
    )

    assert response.status_code == 200
    arguments = captured["payload"]["messages"][0][1]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert arguments == {"filePath": "/tmp/data.csv"}


def test_model_bridge_caps_tokens_per_call() -> None:
    captured: dict[str, Any] = {}

    async def fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return _upstream("OK")

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
            max_tokens_per_call=64,
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fake_chat,
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "inspect"}],
            "max_tokens": 8192,
        },
    )

    assert response.status_code == 200
    assert captured["payload"]["max_tokens"] == 64
    assert captured["payload"]["temperature"] == 1.0
    assert "seed" not in captured["payload"]["generation_kwargs"]
    assert (
        client.get(
            f"/v1/benchflow/logprobs/{response.json()['id']}",
        ).status_code
        == 404
    )


def test_model_bridge_fixes_seed_for_tool_evaluation_only() -> None:
    captured: dict[str, Any] = {}

    async def fake_chat(payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return _upstream("OK")

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fake_chat,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "inspect"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {"type": "object"}},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured["payload"]["temperature"] == 1.0
    assert captured["payload"]["generation_kwargs"]["seed"] == 0


def test_model_bridge_retains_eight_concurrent_sixty_five_turn_rollouts() -> None:
    async def fake_chat(_payload: dict[str, Any]) -> dict[str, Any]:
        return _upstream("A")

    app = create_model_bridge_app(
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
        ),
        tokenizer=FakeTokenizer(),
        chat_call=fake_chat,
    )
    client = TestClient(app)
    completion_ids = []
    for _ in range(8 * 65):
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "inspect"}],
                "logprobs": True,
            },
        )
        completion_ids.append(response.json()["id"])

    assert (
        client.get(
            f"/v1/benchflow/logprobs/{completion_ids[0]}",
        ).status_code
        == 200
    )


def test_model_bridge_rejects_invalid_token_cap() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
            max_tokens_per_call=0,
        )

    with pytest.raises(ValueError, match="max_sidecar_entries"):
        ModelBridgeConfig(
            upstream_url="http://127.0.0.1:8000",
            tokenizer_id="Qwen/Qwen3-4B",
            max_sidecar_entries=0,
        )
