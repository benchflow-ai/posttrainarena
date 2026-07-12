from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.grpo import (
    OpenCodeRolloutCollector,
    build_grpo_rows,
    sync_checkpoint_to_vllm,
    sync_model_to_vllm,
    task_handle,
    task_id_from_prompt,
    train_grpo,
    trajectory_to_rollout_tokens,
    verifier_reward,
)


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
) -> dict[str, Any]:
    return {
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
            }
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


def test_trajectory_to_rollout_tokens_rejects_history_drift(tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError, match="does not extend"):
        trajectory_to_rollout_tokens(
            path,
            FakeTokenizer(),
            max_completion_tokens=1000,
        )


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
        def to(self, **kwargs):
            captured["model_to"] = kwargs

    class FakeProcessor:
        def save_pretrained(self, path):
            captured["processor_path"] = path

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.model = FakeModel()
            self.processing_class = FakeProcessor()

        def train(self):
            return SimpleNamespace(metrics={"loss": 0.5})

        def save_model(self, path):
            captured["model_path"] = path

    monkeypatch.setattr(trl, "GRPOConfig", FakeConfig)
    monkeypatch.setattr(trl, "GRPOTrainer", FakeTrainer)
    monkeypatch.setattr(
        grpo_module,
        "supported_kwargs",
        lambda _callable, values: values,
    )
    monkeypatch.setenv("TRL_VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000")
    output_dir = tmp_path / "checkpoint"

    payload = train_grpo(
        config=config,
        model=config.model,
        tasks_dir=tasks_dir,
        task_ids=["task-a"],
        jobs_dir=tmp_path / "jobs",
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
    assert "environment_factory" not in trainer_kwargs
    assert args["use_vllm"] is True
    assert args["vllm_mode"] == "server"
    assert args["vllm_server_base_url"] == "http://127.0.0.1:8000"
    assert args["vllm_importance_sampling_correction"] is True
    assert payload["rollout_contract"]["endpoint_sync"] == ("trl-vllm-sync-weights")


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
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
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
    assert captured["sync_kwargs"]["config"] is config
    assert payload["synced"] is True
