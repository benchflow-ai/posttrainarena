from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.grpo import (
    CollectedRollout,
    OpenCodeRolloutCollector,
    RolloutTokens,
    build_grpo_rows,
    sync_checkpoint_to_vllm,
    sync_model_to_vllm,
    sync_reference_to_vllm,
    task_handle,
    task_id_from_prompt,
    train_grpo,
    trajectory_to_rollout_tokens,
    verifier_reward,
)
from posttrainarena.benchflow_pipeline.io import write_json


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    @staticmethod
    def _content(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        return json.dumps(content, sort_keys=True, separators=(",", ":"))

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        assert tokenize is True
        text = ""
        if tools:
            text += f"<tools>{json.dumps(tools, sort_keys=True)}</tools>"
        for message in messages:
            role = str(message["role"])
            text += f"<{role}>{self._content(message)}</{role}>"
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text, add_special_tokens=False)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def _exchange(
    messages: list[dict[str, Any]],
    text: str,
    logprob: float,
    *,
    call_purpose: str = "agent",
) -> dict[str, Any]:
    return {
        "metadata": {
            "call_purpose": call_purpose,
        },
        "request": {
            "body": {
                "messages": messages,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "shell"},
                    }
                ],
            }
        },
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "logprobs": {
                            "content": [
                                {
                                    "token": text,
                                    "bytes": list(text.encode("utf-8")),
                                    "logprob": logprob,
                                }
                            ]
                        },
                    }
                ]
            },
        },
    }


def _write_trajectory(path: Path) -> None:
    first_messages = [{"role": "user", "content": "hi"}]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "ok", "tool_call_id": "call-1"},
    ]
    rows = [
        _exchange(first_messages, "A", -0.1),
        _exchange(second_messages, "B", -0.2),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_task_handles_are_reversible() -> None:
    assert task_handle("task-a") == "benchflow-task://task-a"
    assert task_id_from_prompt("benchflow-task://task-a") == "task-a"

    with pytest.raises(ValueError, match="Unexpected GRPO prompt"):
        task_id_from_prompt("plain prompt")


def test_build_grpo_rows_requires_snapshotted_tasks(tmp_path: Path) -> None:
    (tmp_path / "task-a").mkdir()

    assert build_grpo_rows(tmp_path, ["task-a"]) == [
        {
            "prompt": "benchflow-task://task-a",
            "benchflow_task_id": "task-a",
        }
    ]

    with pytest.raises(FileNotFoundError):
        build_grpo_rows(tmp_path, ["missing"])


def test_trajectory_to_rollout_tokens_masks_environment_feedback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    _write_trajectory(path)

    tokens = trajectory_to_rollout_tokens(
        path,
        FakeTokenizer(),
        max_completion_tokens=1000,
    )

    assert tokens.logprobs[0] == -0.1
    assert tokens.logprobs[-1] == -0.2
    assert tokens.env_mask[0] == 1
    assert tokens.env_mask[-1] == 1
    assert 0 in tokens.env_mask
    assert len(tokens.completion_ids) == len(tokens.logprobs) == len(tokens.env_mask)
    assert sum(tokens.env_mask) == 2


def test_trajectory_to_rollout_tokens_ignores_helper_calls(tmp_path: Path) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    helper = _exchange(
        [{"role": "user", "content": "generate a title"}],
        "Task title",
        -0.5,
        call_purpose="title",
    )
    agent = _exchange([{"role": "user", "content": "hi"}], "A", -0.1)
    path.write_text(json.dumps(helper) + "\n" + json.dumps(agent) + "\n")

    tokens = trajectory_to_rollout_tokens(
        path,
        FakeTokenizer(),
        max_completion_tokens=1000,
    )

    assert tokens.completion_ids == [ord("A")]
    assert tokens.logprobs == [-0.1]


def test_trajectory_to_rollout_tokens_ignores_failed_provider_attempts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    failed = _exchange([{"role": "user", "content": "hi"}], "ignored", -0.5)
    failed["response"] = {
        "status_code": 500,
        "body": {"error": {"message": "upstream failed"}},
    }
    agent = _exchange([{"role": "user", "content": "hi"}], "A", -0.1)
    path.write_text(json.dumps(failed) + "\n" + json.dumps(agent) + "\n")

    tokens = trajectory_to_rollout_tokens(
        path,
        FakeTokenizer(),
        max_completion_tokens=1000,
    )

    assert tokens.completion_ids == [ord("A")]
    assert tokens.logprobs == [-0.1]


def test_trajectory_to_rollout_tokens_requires_call_purpose(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    row = _exchange([{"role": "user", "content": "hi"}], "A", -0.1)
    row.pop("metadata")
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError, match="call-purpose metadata"):
        trajectory_to_rollout_tokens(
            path,
            FakeTokenizer(),
            max_completion_tokens=1000,
        )


def test_trajectory_to_rollout_tokens_masks_canonicalized_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    first_messages = [{"role": "user", "content": "hi"}]
    rows = [
        _exchange(first_messages, "A", -0.1),
        _exchange(
            [
                *first_messages,
                {"role": "assistant", "content": "different"},
            ],
            "B",
            -0.2,
        ),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    tokens = trajectory_to_rollout_tokens(
        path,
        FakeTokenizer(),
        max_completion_tokens=1000,
    )

    assert sum(tokens.env_mask) == 1
    assert tokens.logprobs[-1] == -0.2
    assert 0 in tokens.env_mask


def test_trajectory_to_rollout_tokens_resets_on_original_prompt_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    rows = [
        _exchange([{"role": "user", "content": "hi"}], "A", -0.1),
        _exchange([{"role": "user", "content": "bye"}], "B", -0.2),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    tokens = trajectory_to_rollout_tokens(
        path,
        FakeTokenizer(),
        max_completion_tokens=1000,
    )

    expected_prompt = FakeTokenizer().apply_chat_template(
        [{"role": "user", "content": "bye"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "shell"},
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert tokens.prompt_ids == expected_prompt
    assert tokens.completion_ids == [ord("B")]
    assert tokens.logprobs == [-0.2]
    assert tokens.env_mask == [1]


def test_trajectory_to_rollout_tokens_requires_provider_logprobs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    row = _exchange([{"role": "user", "content": "hi"}], "A", -0.1)
    row["response"]["body"]["choices"][0]["logprobs"] = None
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError, match="no sampled-token logprobs"):
        trajectory_to_rollout_tokens(
            path,
            FakeTokenizer(),
            max_completion_tokens=1000,
        )


def test_trajectory_to_rollout_tokens_resolves_streaming_logprobs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "llm_trajectory.jsonl"
    row = _exchange([{"role": "user", "content": "hi"}], "A", -0.1)
    row["response"]["body"]["id"] = "chatcmpl-1"
    expected = row["response"]["body"]["choices"][0].pop("logprobs")
    path.write_text(json.dumps(row) + "\n")

    tokens = trajectory_to_rollout_tokens(
        path,
        FakeTokenizer(),
        max_completion_tokens=1000,
        logprob_resolver=lambda completion_id: (
            expected if completion_id == "chatcmpl-1" else {}
        ),
    )

    assert tokens.completion_ids == [ord("A")]
    assert tokens.logprobs == [-0.1]


def test_verifier_reward_uses_rollout_metadata() -> None:
    assert verifier_reward(["a", "b"], rollout_reward=[1.0, 0.25]) == [1.0, 0.25]

    with pytest.raises(RuntimeError, match="missing verifier rewards"):
        verifier_reward(["a"], rollout_reward=None)


def test_collector_runs_one_opencode_rollout_per_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "task-a").mkdir(parents=True)
    calls: list[dict[str, Any]] = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        rollout_dir = kwargs["jobs_dir"] / "job" / "task-a"
        _write_trajectory(rollout_dir / "trajectory" / "llm_trajectory.jsonl")
        return {
            "health": {
                "rows": [
                    {
                        "reward": 1.0,
                        "rollout_dir": str(rollout_dir),
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.grpo.evaluate",
        fake_evaluate,
    )
    collector = OpenCodeRolloutCollector(
        config=config,
        model="/tmp/student",
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
    )
    trainer = SimpleNamespace(
        processing_class=FakeTokenizer(),
        state=SimpleNamespace(global_step=3),
        accelerator=SimpleNamespace(process_index=0),
    )

    output = collector(
        [task_handle("task-a"), task_handle("task-a")],
        trainer,
    )

    assert len(calls) == 2
    assert all(call["model_role"] == "student" for call in calls)
    assert all(call["capture_token_logprobs"] is True for call in calls)
    assert output["rollout_reward"] == [1.0, 1.0]
    assert [sum(mask) for mask in output["env_mask"]] == [2, 2]
    assert len(collector.records) == 2


def test_collector_parallelizes_rollouts_up_to_harness_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        harness=replace(config.harness, concurrency=2),
    )
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "task-a").mkdir(parents=True)
    collector = OpenCodeRolloutCollector(
        config=config,
        model="/tmp/student",
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
    )
    barrier = threading.Barrier(2)
    indexes: list[int] = []

    def fake_collect_one(
        *,
        rollout_index: int,
        task_id: str,
        tokenizer: Any,
        trainer: Any,
    ) -> CollectedRollout:
        del tokenizer, trainer
        indexes.append(rollout_index)
        barrier.wait(timeout=2)
        return CollectedRollout(
            task_id=task_id,
            reward=1.0,
            rollout_dir=tmp_path / f"rollout-{rollout_index}",
            tokens=RolloutTokens(
                prompt_ids=[1],
                completion_ids=[2],
                logprobs=[-0.1],
                env_mask=[1],
            ),
        )

    monkeypatch.setattr(collector, "_collect_one", fake_collect_one)
    trainer = SimpleNamespace(processing_class=object())

    output = collector(
        [task_handle("task-a"), task_handle("task-a")],
        trainer,
    )

    assert sorted(indexes) == [0, 1]
    assert output["benchflow_task_id"] == ["task-a", "task-a"]


def test_collector_recollects_instead_of_reusing_stale_rollout_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "task-a").mkdir(parents=True)
    jobs_dir = tmp_path / "jobs"
    attempt_root = jobs_dir / "step-000002/rank-00/rollout-000000/attempt-01"
    rollout_dir = attempt_root / "jobs/job/task-a"
    _write_trajectory(rollout_dir / "trajectory/llm_trajectory.jsonl")
    write_json(
        attempt_root / "metrics.json",
        {
            "health": {
                "rows": [
                    {
                        "reward": 0.75,
                        "rollout_dir": str(rollout_dir),
                    }
                ]
            }
        },
    )
    write_json(attempt_root / "rollout_error.json", {"error": "stale"})

    calls = 0

    def fresh_evaluate(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "health": {
                "rows": [
                    {
                        "reward": 1.0,
                        "rollout_dir": str(rollout_dir),
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.grpo.evaluate",
        fresh_evaluate,
    )
    collector = OpenCodeRolloutCollector(
        config=config,
        model="/tmp/student",
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
    )
    trainer = SimpleNamespace(
        processing_class=FakeTokenizer(),
        state=SimpleNamespace(global_step=2),
        accelerator=SimpleNamespace(process_index=0),
    )

    output = collector([task_handle("task-a")], trainer)

    assert calls == 1
    assert output["rollout_reward"] == [1.0]
    assert (attempt_root / "grpo_tokens.json").is_file()
    assert not (attempt_root / "rollout_error.json").exists()


def test_collector_uses_local_bridge_control_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    captured: dict[str, Any] = {}

    class FakeResponse:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, Any]:
            return {"logprobs": {"content": [{"token": "A"}]}}

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setenv(
        "BENCHFLOW_PROVIDER_BASE_URL",
        "https://public.example/v1",
    )
    monkeypatch.setenv(
        "BENCHFLOW_MODEL_BRIDGE_CONTROL_URL",
        "http://127.0.0.1:8002/v1",
    )
    monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "secret")
    collector = OpenCodeRolloutCollector(
        config=config,
        model="/tmp/student",
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
    )

    payload = collector._resolve_bridge_logprobs("chatcmpl-1")

    assert payload["content"]
    assert captured["url"] == ("http://127.0.0.1:8002/v1/benchflow/logprobs/chatcmpl-1")
    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer secret"}


def test_collector_retries_failed_opencode_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "task-a").mkdir(parents=True)
    attempts = 0

    def fake_evaluate(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient endpoint failure")
        rollout_dir = kwargs["jobs_dir"] / "job" / "task-a"
        _write_trajectory(rollout_dir / "trajectory" / "llm_trajectory.jsonl")
        return {
            "health": {
                "rows": [
                    {
                        "reward": 0.5,
                        "rollout_dir": str(rollout_dir),
                    }
                ]
            }
        }

    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.grpo.evaluate",
        fake_evaluate,
    )
    collector = OpenCodeRolloutCollector(
        config=config,
        model="/tmp/student",
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
    )
    trainer = SimpleNamespace(
        processing_class=FakeTokenizer(),
        state=SimpleNamespace(global_step=1),
        accelerator=SimpleNamespace(process_index=0),
    )

    output = collector([task_handle("task-a")], trainer)

    assert attempts == 2
    assert output["rollout_reward"] == [0.5]
    assert list((tmp_path / "jobs").rglob("rollout_error.json"))


def test_train_grpo_wires_custom_rollout_and_vllm_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import peft
    import transformers
    import trl
    import posttrainarena.benchflow_pipeline.grpo as grpo_module

    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "task-a").mkdir(parents=True)
    captured: dict[str, Any] = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeModel:
        pass

    class FakeMerged:
        def save_pretrained(self, path, **kwargs):
            captured["merged_save"] = (path, kwargs)

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path):
            captured["adapter_load"] = (model, path)
            return cls()

        def merge_and_unload(self):
            return FakeMerged()

    class FakeProcessor:
        def save_pretrained(self, path):
            captured.setdefault("processor_paths", []).append(path)

    class FakeVllmClient:
        def close_communicator(self):
            captured["trainer_communicator_closed"] = True

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.model = FakeModel()
            self.processing_class = FakeProcessor()
            self.vllm_generation = SimpleNamespace(vllm_client=FakeVllmClient())

        def train(self):
            return SimpleNamespace(metrics={"loss": 0.5})

        def save_model(self, path):
            captured["model_path"] = path

    monkeypatch.setattr(trl, "GRPOConfig", FakeConfig)
    monkeypatch.setattr(trl, "GRPOTrainer", FakeTrainer)
    monkeypatch.setattr(peft, "LoraConfig", FakeConfig)
    monkeypatch.setattr(peft, "PeftModel", FakePeftModel)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: (
            captured.update(base_load=(args, kwargs)) or FakeModel()
        ),
    )
    monkeypatch.setattr(
        grpo_module,
        "supported_kwargs",
        lambda _callable, values: values,
    )
    tokenizer = object()
    monkeypatch.setattr(
        grpo_module,
        "_load_tokenizer",
        lambda config, model: tokenizer,
    )
    monkeypatch.setenv("TRL_VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000")
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "checkpoint"

    payload = train_grpo(
        config=config,
        model=config.model,
        tasks_dir=tasks_dir,
        task_ids=["task-a"],
        jobs_dir=tmp_path / "jobs",
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        run_name="test-run",
    )

    trainer_kwargs = captured["trainer_kwargs"]
    args = trainer_kwargs["args"].values
    assert isinstance(
        trainer_kwargs["rollout_func"],
        OpenCodeRolloutCollector,
    )
    assert trainer_kwargs["reward_funcs"] == [verifier_reward]
    assert trainer_kwargs["processing_class"] is tokenizer
    assert trainer_kwargs["peft_config"].values == {
        "r": config.grpo.lora_r,
        "lora_alpha": config.grpo.lora_alpha,
        "lora_dropout": config.grpo.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
    }
    assert "environment_factory" not in trainer_kwargs
    assert args["use_vllm"] is True
    assert args["vllm_mode"] == "server"
    assert args["vllm_server_base_url"] == "http://127.0.0.1:8000"
    assert args["vllm_importance_sampling_correction"] is True
    assert args["log_completions"] is False
    assert args["generation_batch_size"] == 2
    assert args["max_steps"] == config.grpo.max_steps
    assert "num_train_epochs" not in args
    assert args["gradient_checkpointing"] is True
    assert captured["trainer_communicator_closed"] is True
    assert captured["model_path"] == str(adapter_dir)
    assert captured["adapter_load"][1] == str(adapter_dir)
    assert captured["merged_save"] == (
        str(output_dir),
        {"safe_serialization": True},
    )
    assert payload["adapter_dir"] == str(adapter_dir)
    assert payload["merged_model_dir"] == str(output_dir)
    assert payload["quantization"] is None
    dependency = json.loads((adapter_dir / "adapter_dependency.json").read_text())
    assert dependency["base_checkpoint"] == config.model
    assert dependency["published_base_sibling"] == "../sft-merged"
    assert payload["rollout_contract"]["endpoint_sync"] == ("trl-vllm-sync-weights")

    full_config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-full.toml")
    full_payload = train_grpo(
        config=full_config,
        model=full_config.model,
        tasks_dir=tasks_dir,
        task_ids=["task-a"],
        jobs_dir=tmp_path / "full-jobs",
        adapter_dir=tmp_path / "full-adapter",
        output_dir=tmp_path / "full-checkpoint",
        run_name="full-run",
    )
    full_args = captured["trainer_kwargs"]["args"].values

    assert full_args["num_train_epochs"] == 1.0
    assert "max_steps" not in full_args
    assert full_args["generation_batch_size"] == 16
    assert full_payload["num_train_epochs"] == 1.0
    assert full_payload["max_steps"] is None
    assert full_payload["generation_batch_size"] == 16


def test_sync_model_to_vllm_closes_weight_communicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trl.generation.vllm_generation as vllm_module

    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    captured: dict[str, Any] = {}

    class FakeClient:
        def close_communicator(self):
            captured["closed"] = True

    class FakeGeneration:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.vllm_client = FakeClient()

        def sync_weights(self):
            captured["synced"] = True

    monkeypatch.setattr(vllm_module, "VLLMGeneration", FakeGeneration)
    monkeypatch.setenv("TRL_VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000")
    model = object()
    accelerator = object()
    tokenizer = object()

    sync_model_to_vllm(
        config=config,
        model=model,
        accelerator=accelerator,
        processing_class=tokenizer,
    )

    assert captured["synced"] is True
    assert captured["closed"] is True
    assert captured["kwargs"]["model"] is model
    assert captured["kwargs"]["accelerator"] is accelerator
    assert captured["kwargs"]["processing_class"] is tokenizer
    assert captured["kwargs"]["mode"] == "server"


def test_sync_checkpoint_loads_saved_policy_before_endpoint_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate
    import transformers
    import posttrainarena.benchflow_pipeline.grpo as grpo_module

    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured: dict[str, Any] = {}

    class FakeAccelerator:
        device = "cuda:0"

        def __init__(self, **kwargs):
            captured["accelerator_kwargs"] = kwargs

    class FakeTokenizer:
        pass

    class FakeModel:
        def to(self, device):
            captured["model_device"] = device

    monkeypatch.setattr(accelerate, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: (
            captured.update(tokenizer_args=args, tokenizer_kwargs=kwargs)
            or FakeTokenizer()
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: (
            captured.update(model_args=args, model_kwargs=kwargs) or FakeModel()
        ),
    )
    monkeypatch.setattr(
        grpo_module,
        "sync_model_to_vllm",
        lambda **kwargs: captured.update(sync_kwargs=kwargs),
    )

    payload = sync_checkpoint_to_vllm(
        config=config,
        checkpoint=checkpoint,
    )

    assert captured["accelerator_kwargs"] == {"mixed_precision": "bf16"}
    assert captured["model_device"] == "cuda:0"
    assert captured["tokenizer_args"] == (str(checkpoint),)
    assert captured["tokenizer_kwargs"] == {
        "trust_remote_code": True,
        "fix_mistral_regex": True,
    }
    assert captured["sync_kwargs"]["config"] is config
    assert payload["synced"] is True

    sync_reference_to_vllm(
        config=config,
        reference=config.model,
    )

    assert captured["model_args"] == (config.model,)
    assert captured["model_kwargs"]["revision"] == config.model_revision
