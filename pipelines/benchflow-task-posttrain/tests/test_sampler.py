from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.grpo import (
    OpenCodeRolloutCollector,
    reward_group_diagnostics,
    task_handle,
    train_grpo,
    trajectory_to_rollout_tokens,
    verifier_reward,
)
from posttrainarena.benchflow_pipeline.pipeline import Pipeline
from posttrainarena.benchflow_pipeline.sampler import (
    coverage_problem,
    cover_schedule,
    sampler_plan,
    sampler_report,
    train_task_stats,
)

from test_grpo import FakeTokenizer, _exchange, _write_trajectory


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"


def _cover_config(**grpo: Any):
    """The smoke recipe (2 generations per group) with 2 task groups per step under cover."""
    config = load_config(SMOKE)
    values = {
        "task_sampler": "cover",
        "max_steps": 8,
        "generation_batch_size": 4,
        "gradient_accumulation_steps": 4,
        **grpo,
    }
    config = replace(config, grpo=replace(config.grpo, **values))
    config.validate()
    return config


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("task_count", [1, 3, 7, 10, 33])
@pytest.mark.parametrize("per_step", [1, 2, 3, 8, 16])
def test_cover_schedule_reaches_every_task_with_balanced_exposure(
    seed: int, task_count: int, per_step: int
) -> None:
    tasks = [f"task-{index:02d}" for index in range(task_count)]
    steps = math.ceil(task_count / per_step) * 3 + 2

    schedule = cover_schedule(tasks, prompts_per_step=per_step, steps=steps, seed=seed)

    assert len(schedule) == steps
    assert all(len(step) == per_step for step in schedule)
    first_pass = math.ceil(task_count / per_step)
    assert {task for step in schedule[:first_pass] for task in step} == set(tasks)
    counts: Counter[str] = Counter()
    for step in schedule:
        if per_step <= task_count:
            assert len(set(step)) == per_step
        counts.update(step)
        assert max(counts[t] for t in tasks) - min(counts[t] for t in tasks) <= 1
    assert schedule == cover_schedule(tasks, prompts_per_step=per_step, steps=steps, seed=seed)


def test_cover_schedule_depends_on_seed() -> None:
    tasks = [f"task-{index}" for index in range(20)]

    first = cover_schedule(tasks, prompts_per_step=4, steps=5, seed=1)
    second = cover_schedule(tasks, prompts_per_step=4, steps=5, seed=2)

    assert first != second


def test_sampler_plan_records_the_fixed_cover_schedule() -> None:
    config = _cover_config()
    tasks = [f"task-{index}" for index in range(5)]

    plan = sampler_plan(config, tasks)

    assert plan["sampler"] == "cover"
    assert plan["seed"] == 42
    assert plan["prompts_per_step"] == 2
    assert len(plan["planned_steps"]) == 8
    assert plan["planned_exposures"]["covered_task_count"] == 5
    assert plan["planned_exposures"]["steps_to_full_coverage"] == 3
    assert plan["planned_exposures"]["max_exposures"] - plan["planned_exposures"]["min_exposures"] <= 1


def test_trl_sampler_plan_has_no_schedule() -> None:
    config = load_config(SMOKE)

    plan = sampler_plan(config, ["a", "b"])

    assert plan["sampler"] == "trl"
    assert plan["planned_steps"] is None
    assert "last partial chunk" in plan["description"]


@pytest.mark.parametrize(
    ("grpo", "message"),
    [
        ({"task_sampler": "cover", "max_steps": None}, "requires grpo.max_steps"),
        ({"task_sampler": "cover", "max_steps": 2, "gradient_accumulation_steps": 1}, "one generation batch"),
        ({"task_sampler": "random"}, "trl or cover"),
        ({"rollout_failure_policy": "skip"}, "raise or mask"),
        ({"max_masked_rollout_fraction": 1.0}, "max_masked_rollout_fraction"),
        ({"seed": -1}, "grpo.seed"),
    ],
)
def test_config_rejects_invalid_sampler_values(grpo: dict[str, Any], message: str) -> None:
    config = load_config(SMOKE)
    config = replace(config, grpo=replace(config.grpo, **grpo))

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_pipeline_refuses_a_cover_budget_that_misses_tasks(tmp_path: Path) -> None:
    # 15 training tasks, 2 per step: 7 steps reach 14, 8 steps reach all 15.
    short = replace(_cover_config(max_steps=7), output_root=tmp_path)
    enough = replace(_cover_config(max_steps=8), output_root=tmp_path)

    with pytest.raises(ValueError, match="cannot cover the 15 training tasks"):
        Pipeline(short, run_name="short", dry_run=True)
    Pipeline(enough, run_name="enough", dry_run=True)
    partial = replace(short, grpo=replace(short.grpo, require_full_coverage=False))
    assert coverage_problem(partial, ["t"] * 1) is None
    Pipeline(partial, run_name="partial", dry_run=True)


def test_train_task_stats_counts_every_task_and_masks_infra_errors() -> None:
    records = [
        {"task_id": "a", "group_index": 0, "global_step": 0, "reward": reward}
        for reward in (1.0, 0.0, 1.0)
    ] + [
        {"task_id": "b", "group_index": 1, "global_step": 0, "reward": 0.0, "truncated": True},
        {"task_id": "b", "group_index": 1, "global_step": 0, "reward": 0.0},
    ]
    masked = [
        {"task_id": "a", "group_index": 0, "global_step": 0, "kind": "infra"},
        {"task_id": "b", "group_index": 1, "global_step": 0, "kind": "trajectory"},
        {"task_id": "b", "group_index": 1, "global_step": 0, "kind": "infra"},
    ]
    failed = [
        {"task_id": "a", "attempt": 1, "kind": "infra"},
        {"task_id": "a", "attempt": 2, "kind": "infra"},
    ]

    stats = train_task_stats(
        ["a", "b", "c"],
        records=records,
        masked_records=masked,
        failed_attempts=failed,
        num_generations=4,
    )

    a, b, c = (stats["tasks"][task] for task in ("a", "b", "c"))
    assert (a["rollouts"], a["scored_rollouts"], a["passes"]) == (4, 3, 2)
    assert a["reward_mean"] == pytest.approx(2 / 3)
    assert a["reward_variance"] == pytest.approx(2 / 9)
    assert (a["infra_errors"], a["trajectory_errors"], a["failed_attempts"]) == (1, 0, 2)
    assert a["groups_with_variance"] == 1
    assert a["sampled_steps"] == [0]
    assert (b["infra_errors"], b["trajectory_errors"], b["truncated_rollouts"]) == (1, 1, 1)
    assert b["all_fail_groups"] == 1
    assert c["rollouts"] == 0 and c["pass_rate"] is None
    assert stats["unsampled_task_ids"] == ["c"]
    assert stats["sampled_task_count"] == 2
    assert stats["totals"]["rollouts"] == 8
    assert stats["totals"]["masked_rollout_fraction"] == pytest.approx(3 / 8)


def test_reward_groups_count_masked_rollouts_toward_completeness() -> None:
    records = [
        {"task_id": "a", "group_index": 0, "global_step": 0, "reward": reward}
        for reward in (1.0, 0.0, 1.0)
    ]
    masked = [{"task_id": "a", "group_index": 0, "global_step": 0}]

    diagnostics = reward_group_diagnostics(records, num_generations=4, masked_records=masked)

    group = diagnostics["groups"][0]
    assert group["complete"] is True
    assert group["has_variance"] is True
    assert group["masked_count"] == 1
    assert diagnostics["masked_rollout_count"] == 1
    assert diagnostics["incomplete_group_count"] == 0


def test_verifier_reward_passes_masked_rollouts_as_none() -> None:
    assert verifier_reward(["a", "b"], rollout_reward=[None, 1.0]) == [None, 1.0]


def test_trajectory_truncation_keeps_the_prefix(tmp_path: Path) -> None:
    path = tmp_path / "trajectory/llm_trajectory.jsonl"
    _write_trajectory(path)
    full = trajectory_to_rollout_tokens(path, FakeTokenizer(), max_completion_tokens=10_000)

    with pytest.raises(RuntimeError, match="limit is 5"):
        trajectory_to_rollout_tokens(path, FakeTokenizer(), max_completion_tokens=5)
    tokens = trajectory_to_rollout_tokens(
        path, FakeTokenizer(), max_completion_tokens=5, truncate=True
    )

    assert tokens.truncated_from == len(full.completion_ids)
    assert tokens.completion_ids == full.completion_ids[:5]
    assert tokens.env_mask == full.env_mask[:5]
    assert full.truncated_from is None


def _collector(tmp_path: Path, config: Any) -> OpenCodeRolloutCollector:
    tasks_dir = tmp_path / "tasks"
    for task_id in ("task-a", "task-b"):
        (tasks_dir / task_id).mkdir(parents=True, exist_ok=True)
    return OpenCodeRolloutCollector(
        config=config,
        model="/tmp/student",
        tasks_dir=tasks_dir,
        jobs_dir=tmp_path / "jobs",
        task_ids=["task-a", "task-b"],
    )


def _trainer(step: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        processing_class=FakeTokenizer(),
        state=SimpleNamespace(global_step=step),
        accelerator=SimpleNamespace(process_index=0),
    )


def test_mask_policy_masks_failed_rollouts_and_logs_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _cover_config(rollout_failure_policy="mask", max_masked_rollout_fraction=0.9)

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        if kwargs["task_ids"] == ["task-b"]:
            raise RuntimeError("Daytona sandbox create failed")
        rollout_dir = kwargs["jobs_dir"] / "job" / "task-a"
        _write_trajectory(rollout_dir / "trajectory" / "llm_trajectory.jsonl")
        return {"health": {"rows": [{"reward": 1.0, "rollout_dir": str(rollout_dir)}]}}

    monkeypatch.setattr("posttrainarena.benchflow_pipeline.grpo.evaluate", fake_evaluate)
    collector = _collector(tmp_path, config)

    output = collector(
        [task_handle("task-a")] * 2 + [task_handle("task-b")] * 2, _trainer(step=3)
    )

    assert output["rollout_reward"] == [1.0, 1.0, None, None]
    assert output["env_mask"][2] == [0]
    assert [record["kind"] for record in collector.masked_records] == ["infra", "infra"]
    assert len(collector.failed_attempts) == 2 * config.grpo.rollout_attempts
    logged = [
        json.loads(line)
        for line in (tmp_path / "jobs/sampled_steps.jsonl").read_text().splitlines()
    ]
    assert logged[0]["global_step"] == 3
    assert logged[0]["task_ids"] == ["task-a", "task-b"]
    assert logged[0]["masked_rollouts"] == 2
    stats = json.loads((tmp_path / "jobs/train_task_stats.json").read_text())
    assert stats["tasks"]["task-b"]["infra_errors"] == 2
    assert stats["tasks"]["task-b"]["scored_rollouts"] == 0
    assert stats["tasks"]["task-a"]["passes"] == 2


def test_mask_policy_stops_when_a_whole_batch_or_too_many_rollouts_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(**_: Any) -> dict[str, Any]:
        raise RuntimeError("sandbox create failed")

    monkeypatch.setattr("posttrainarena.benchflow_pipeline.grpo.evaluate", failing)
    config = _cover_config(rollout_failure_policy="mask", max_masked_rollout_fraction=0.9)
    with pytest.raises(RuntimeError, match="Every rollout in GRPO generation batch 0"):
        _collector(tmp_path / "all", config)([task_handle("task-a")] * 2, _trainer())

    calls = 0

    def half(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if kwargs["task_ids"] == ["task-b"]:
            raise RuntimeError("sandbox create failed")
        rollout_dir = kwargs["jobs_dir"] / "job" / "task-a"
        _write_trajectory(rollout_dir / "trajectory" / "llm_trajectory.jsonl")
        return {"health": {"rows": [{"reward": 0.0, "rollout_dir": str(rollout_dir)}]}}

    monkeypatch.setattr("posttrainarena.benchflow_pipeline.grpo.evaluate", half)
    strict = _cover_config(rollout_failure_policy="mask", max_masked_rollout_fraction=0.25)
    with pytest.raises(RuntimeError, match="above grpo.max_masked_rollout_fraction"):
        _collector(tmp_path / "strict", strict)(
            [task_handle("task-a"), task_handle("task-b")], _trainer()
        )


def test_errored_health_row_is_masked_not_scored_zero_under_mask_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def errored(**kwargs: Any) -> dict[str, Any]:
        rollout_dir = kwargs["jobs_dir"] / "job" / "task-a"
        _write_trajectory(rollout_dir / "trajectory" / "llm_trajectory.jsonl")
        row = {
            "task_id": "task-a",
            "reward": 0.0,
            "scored": True,
            "error": "Daytona transport error",
            "verifier_error": None,
            "valid_llm_trajectory": True,
            "rollout_dir": str(rollout_dir),
        }
        if kwargs["task_ids"] == ["task-b"]:
            row = {**row, "task_id": "task-b", "error": "agent wall-clock budget exceeded", "error_category": "timeout"}
        return {"health": {"rows": [row]}}

    monkeypatch.setattr("posttrainarena.benchflow_pipeline.grpo.evaluate", errored)
    masked = _collector(tmp_path / "mask", _cover_config(rollout_failure_policy="mask", max_masked_rollout_fraction=0.9))
    output = masked([task_handle("task-a"), task_handle("task-b")], _trainer())

    # The infrastructure error is masked; the timeout is a scored failure.
    assert output["rollout_reward"] == [None, 0.0]
    assert masked.masked_records[0]["kind"] == "infra"

    legacy = _collector(tmp_path / "raise", _cover_config())
    assert legacy([task_handle("task-a")], _trainer())["rollout_reward"] == [0.0]


def test_mask_policy_truncates_long_trajectories_instead_of_dropping_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def long_rollout(**kwargs: Any) -> dict[str, Any]:
        rollout_dir = kwargs["jobs_dir"] / "job" / "task-a"
        path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = _exchange([{"role": "user", "content": "hi"}], "A" * 50, -0.1)
        row["response"]["body"]["choices"][0]["logprobs"]["content"] = [
            {"token": "A", "bytes": [65], "logprob": -0.1} for _ in range(50)
        ]
        path.write_text(json.dumps(row) + "\n")
        return {"health": {"rows": [{"reward": 0.0, "rollout_dir": str(rollout_dir)}]}}

    monkeypatch.setattr("posttrainarena.benchflow_pipeline.grpo.evaluate", long_rollout)
    config = _cover_config(rollout_failure_policy="mask")
    config = replace(config, runtime=replace(config.runtime, max_completion_length=10))
    collector = _collector(tmp_path, config)

    output = collector([task_handle("task-a")], _trainer())

    assert output["rollout_reward"] == [0.0]
    assert len(output["completion_ids"][0]) == 10
    assert collector.records[0]["truncated"] is True
    assert collector.records[0]["completion_tokens_untruncated"] == 50


def test_sampler_report_compares_actual_steps_with_the_plan() -> None:
    config = _cover_config(max_steps=3)
    plan = sampler_plan(config, ["a", "b", "c", "d"])
    batches = [
        {
            "generation_index": index,
            "global_step": index,
            "task_ids": step,
            "groups": [{"group_index": 2 * index + k, "task_id": task} for k, task in enumerate(step)],
            "rollouts": 4,
            "masked_rollouts": 0,
        }
        for index, step in enumerate(plan["planned_steps"])
    ]
    batches[2] = {**batches[2], "groups": [{"group_index": 4, "task_id": "zzz"}]}

    report = sampler_report(plan, batches=batches, trainer_args={"steps_per_generation": 4})

    assert [step["global_step"] for step in report["steps"]] == [0, 1, 2]
    assert report["plan_mismatch_steps"] == [2]
    assert report["trainer"] == {"steps_per_generation": 4}
    assert report["actual_exposures"]["covered_task_count"] <= 4


def test_train_grpo_feeds_the_cover_schedule_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import peft
    import transformers
    import trl
    import posttrainarena.benchflow_pipeline.grpo as grpo_module

    config = _cover_config(
        max_steps=3,
        rollout_failure_policy="mask",
        lora_target_parameters=("mlp.experts.gate_up_proj",),
    )
    tasks = ["task-a", "task-b", "task-c"]
    tasks_dir = tmp_path / "tasks"
    for task_id in tasks:
        (tasks_dir / task_id).mkdir(parents=True)
    captured: dict[str, Any] = {}

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.values = kwargs

    class FakeModel:
        def named_parameters(self):
            import torch

            yield "layer.lora_B.default.weight", torch.tensor([1.0])

        def save_pretrained(self, path: str, **_: Any) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"m")

        def merge_and_unload(self) -> "FakeModel":
            return self

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.model = FakeModel()
            self.processing_class = None
            self.args = SimpleNamespace(**kwargs["args"].values, steps_per_generation=4)
            self.state = SimpleNamespace(log_history=[], global_step=0)
            self.accelerator = SimpleNamespace(process_index=0)
            self.rollout_func = kwargs["rollout_func"]
            self.dataset = kwargs["train_dataset"]

        def train(self) -> Any:
            rows = [row["benchflow_task_id"] for row in self.dataset]
            for step in range(3):
                self.state.global_step = step
                chunk = rows[2 * step : 2 * step + 2]
                start = self.rollout_func._rollout_index
                self.rollout_func._rollout_index += 4
                requests = [(start + k, task) for k, task in enumerate(t for t in chunk for _ in range(2))]
                for index, task in requests:
                    self.rollout_func.records.append(
                        {"rollout_index": index, "group_index": index // 2, "global_step": step, "task_id": task, "reward": float(index % 2)}
                    )
                self.rollout_func._log_batch(requests, [SimpleNamespace(masked=False)] * 4, self)
            return SimpleNamespace(metrics={"train_loss": 0.1})

        def save_model(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"a")

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model: Any, path: str) -> FakeModel:
            return FakeModel()

    monkeypatch.setattr(trl, "GRPOConfig", FakeConfig)
    monkeypatch.setattr(trl, "GRPOTrainer", FakeTrainer)
    monkeypatch.setattr(peft, "LoraConfig", FakeConfig)
    monkeypatch.setattr(peft, "PeftModel", FakePeftModel)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeModel())
    monkeypatch.setattr(grpo_module, "supported_kwargs", lambda _callable, values: values)
    monkeypatch.setattr(grpo_module, "_load_tokenizer", lambda config, model: None)
    monkeypatch.setenv("TRL_VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000")

    payload = train_grpo(
        config=config,
        model=config.model,
        tasks_dir=tasks_dir,
        task_ids=tasks,
        jobs_dir=tmp_path / "jobs",
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "merged",
        run_name="cover",
    )

    plan = sampler_plan(config, tasks)
    args = captured["args"].values
    assert captured["peft_config"].values["target_modules"] == "all-linear"
    assert captured["peft_config"].values["target_parameters"] == ["mlp.experts.gate_up_proj"]
    assert args["seed"] == 42
    assert args["shuffle_dataset"] is False
    assert [row["benchflow_task_id"] for row in captured["train_dataset"]] == [
        task for step in plan["planned_steps"] for task in step
    ]
    sampler = payload["sampler"]
    assert [step["task_ids"] for step in sampler["steps"]] == [
        list(dict.fromkeys(step)) for step in plan["planned_steps"]
    ]
    assert sampler["plan_mismatch_steps"] == []
    assert sampler["trainer"]["steps_per_generation"] == 4
    assert payload["task_stats"]["sampled_task_count"] == 3
    assert payload["training_recipe"]["task_sampler"] == "cover"
    assert (tmp_path / "jobs/train_sampler.json").is_file()
    assert (tmp_path / "jobs/sampler_plan.json").is_file()

    pipeline_config = replace(
        config,
        output_root=tmp_path / "runs",
        grpo=replace(config.grpo, require_full_coverage=False),
    )
    pipeline = Pipeline(pipeline_config, run_name="reports", dry_run=False)
    pipeline.layout.grpo_merged.mkdir(parents=True)
    (pipeline.layout.grpo_merged / "train_metrics.json").write_text(json.dumps(payload))
    pipeline._write_grpo_reports()
    assert json.loads((pipeline.layout.reports / "train_task_stats.json").read_text()) == json.loads(
        json.dumps(payload["task_stats"])
    )
    summary = pipeline._write_score(
        baseline_score=0.0,
        sft_score=None,
        grpo_gate_score=0.0,
        final_score=0.0,
        final_model=str(pipeline.layout.grpo_merged),
        grpo_planned=True,
        grpo_ran=True,
    )
    coverage = summary["grpo_training"]["task_coverage"]
    assert coverage["sampler"] == "cover"
    assert coverage["sampled_task_count"] == 3
