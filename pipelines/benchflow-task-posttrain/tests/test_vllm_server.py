from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from posttrainarena.benchflow_pipeline.vllm_server import (
    llm_worker,
    server_weight_name,
    validate_control_host,
)


class _Qwen35Wrapper:
    language_model = object()


class _TextModel:
    model = object()


def test_server_weight_name_maps_qwen35_text_weights() -> None:
    model = _Qwen35Wrapper()

    assert (
        server_weight_name(model, "model.layers.0.mlp.down_proj.weight")
        == "language_model.model.layers.0.mlp.down_proj.weight"
    )
    assert (
        server_weight_name(model, "lm_head.weight") == "language_model.lm_head.weight"
    )
    assert (
        server_weight_name(model, "language_model.model.norm.weight")
        == "language_model.model.norm.weight"
    )


def test_server_weight_name_leaves_text_only_models_unchanged() -> None:
    assert server_weight_name(_TextModel(), "model.embed_tokens.weight") == (
        "model.embed_tokens.weight"
    )


def test_control_host_rejects_public_bindings() -> None:
    validate_control_host("127.0.0.1")
    validate_control_host("::1")
    validate_control_host("localhost")

    with pytest.raises(ValueError, match="loopback"):
        validate_control_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        validate_control_host("trainer.example.com")


def test_worker_closes_communicator_when_parent_pipe_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeLlm:
        def collective_rpc(self, *, method):
            calls.append(method)

    class FakeConnection:
        def send(self, payload):
            assert payload == {"status": "ready"}

        def recv(self):
            raise EOFError

        def close(self):
            calls.append("connection.close")

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **_: FakeLlm()))
    args = SimpleNamespace(
        model="model",
        revision="revision",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        dtype="auto",
        enable_prefix_caching=True,
        kv_cache_dtype="auto",
        max_model_len=1024,
        trust_remote_code=True,
        vllm_model_impl="auto",
        distributed_executor_backend=None,
        speculative_config=None,
        data_parallel_size=1,
    )

    llm_worker(args, 0, 12345, FakeConnection())

    assert calls == ["close_communicator", "connection.close"]
