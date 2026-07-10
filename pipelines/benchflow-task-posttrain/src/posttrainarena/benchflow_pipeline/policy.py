"""BenchFlow-backed policy evaluation and GRPO training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io import supported_kwargs, write_json


def _spec(config: PipelineConfig, tasks_dir: Path, task_ids: list[str], jobs_dir: Path):
    from benchflow.integrations.trl import BashHarnessConfig, BenchFlowSpec

    return BenchFlowSpec(
        tasks_dir=tasks_dir,
        include_tasks=task_ids,
        bash_harness=BashHarnessConfig(
            environment=config.runtime.environment,
            sandbox_user=config.runtime.sandbox_user,
            jobs_dir=jobs_dir,
            reset_message=(
                "Use run_bash to inspect and solve the task. "
                "Call submit with only the final answer."
            ),
            bash_timeout_sec=config.runtime.bash_timeout_sec,
            max_output_chars=config.runtime.max_output_chars,
        ),
    )


def _common(config: PipelineConfig, output_dir: Path, run_name: str) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "bf16": True,
        "report_to": [config.tracking.report_to]
        if config.tracking.report_to != "none"
        else "none",
        "remove_unused_columns": False,
        "max_completion_length": config.runtime.max_completion_length,
        "max_tool_calling_iterations": config.runtime.max_tool_calling_iterations,
        "log_completions": True,
        "use_vllm": config.runtime.use_vllm,
        "logging_steps": 1,
    }


def _model_init_kwargs(config: PipelineConfig, model: str) -> dict[str, Any] | None:
    if model != config.model:
        return None
    values: dict[str, Any] = {"trust_remote_code": True}
    if config.model_revision:
        values["revision"] = config.model_revision
    return values


def evaluate(
    *,
    config: PipelineConfig,
    model: str,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    output_dir: Path,
    metrics_path: Path,
    run_name: str,
) -> dict[str, Any]:
    from trl import GRPOConfig, GRPOTrainer

    spec = _spec(config, tasks_dir, task_ids, jobs_dir)
    rows = list(spec.train_dataset_rows)
    eval_batch_size = len(rows)
    generation_batch_size = math.lcm(eval_batch_size, config.runtime.num_generations)
    values = {
        **_common(config, output_dir, run_name),
        "do_eval": True,
        "per_device_train_batch_size": config.runtime.num_generations,
        "per_device_eval_batch_size": eval_batch_size,
        "generation_batch_size": generation_batch_size,
        "num_generations": config.runtime.num_generations,
        "num_generations_eval": 1,
    }
    model_init_kwargs = _model_init_kwargs(config, model)
    if model_init_kwargs:
        values["model_init_kwargs"] = model_init_kwargs
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**supported_kwargs(GRPOConfig, values)),
        train_dataset=spec.train_dataset.select(range(1)),
        eval_dataset=spec.train_dataset,
        environment_factory=spec.environment_factory,
        reward_funcs=spec.reward_funcs,
    )
    metrics = trainer.evaluate()
    score = metrics.get("eval_rewards/benchflow_environment_reward/mean")
    if not isinstance(score, int | float):
        score = metrics.get("eval_reward")
    if not isinstance(score, int | float):
        raise RuntimeError(f"Evaluation metrics contain no BenchFlow reward: {metrics}")
    payload = {
        "mode": "eval",
        "model": model,
        "task_ids": [row["benchflow_task_id"] for row in rows],
        "task_count": len(rows),
        "score": float(score),
        "metrics": metrics,
        "jobs_dir": str(jobs_dir),
    }
    write_json(metrics_path, payload)
    return payload


def train_grpo(
    *,
    config: PipelineConfig,
    model: str,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    from trl import GRPOConfig, GRPOTrainer

    spec = _spec(config, tasks_dir, task_ids, jobs_dir)
    values = {
        **_common(config, output_dir, run_name),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": config.grpo.gradient_accumulation_steps,
        "generation_batch_size": config.runtime.num_generations,
        "learning_rate": config.grpo.learning_rate,
        "max_steps": config.grpo.max_steps,
        "save_steps": max(1, config.grpo.max_steps),
        "num_generations": config.runtime.num_generations,
    }
    model_init_kwargs = _model_init_kwargs(config, model)
    if model_init_kwargs:
        values["model_init_kwargs"] = model_init_kwargs
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**supported_kwargs(GRPOConfig, values)),
        **spec.trainer_kwargs(),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None:
        processing_class.save_pretrained(str(output_dir))
    payload = {
        "mode": "grpo",
        "model": model,
        "task_ids": [row["benchflow_task_id"] for row in spec.train_dataset_rows],
        "metrics": result.metrics,
        "jobs_dir": str(jobs_dir),
    }
    write_json(output_dir / "train_metrics.json", payload)
    return payload
