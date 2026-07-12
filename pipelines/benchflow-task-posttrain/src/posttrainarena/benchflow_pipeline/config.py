"""Typed TOML configuration for the public BenchFlow pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


BENCHFLOW_COMMIT = "c441b2abc07f48c03fd6638c5b9bcf7d837b6f38"
GrpoRunPolicy = Literal["on_reward", "always"]
HarnessSkillMode = Literal["no-skill", "with-skill"]
UsageTrackingPolicy = Literal["required"]


def _table(
    data: dict[str, Any], name: str, *, required: bool = False
) -> dict[str, Any]:
    if required and name not in data:
        raise ValueError(f"Missing required [{name}] table")
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    revision: str
    task_list: Path
    path: str = "tasks"


@dataclass(frozen=True)
class RuntimeConfig:
    sandbox: str | None = None
    sandbox_user: str | None = "agent"
    max_completion_length: int = 2048
    num_generations: int = 2


@dataclass(frozen=True)
class HarnessConfig:
    agent: str = "opencode"
    skill_mode: HarnessSkillMode = "no-skill"
    usage_tracking: UsageTrackingPolicy = "required"
    concurrency: int = 1
    sandbox_setup_timeout_sec: int = 300
    agent_idle_timeout_sec: int = 300
    agent_timeout_sec: int = 900
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    base_model_env: str = "BENCHFLOW_BASE_MODEL"
    student_model_env: str = "BENCHFLOW_ADAPTER_MODEL"
    base_url_env: str = "BENCHFLOW_PROVIDER_BASE_URL"
    api_key_env: str = "BENCHFLOW_PROVIDER_API_KEY"


@dataclass(frozen=True)
class TeacherConfig:
    enabled: bool = True
    model: str = "glm/glm-5.1"
    max_attempts: int = 3
    min_verified: int = 1
    min_reward: float = 1.0
    max_accepted_total_tokens: int = 200000
    max_accepted_tool_calls: int = 50


@dataclass(frozen=True)
class SftConfig:
    enabled: bool = True
    max_steps: int = 40
    learning_rate: float = 2e-5
    max_length: int = 4096
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32


@dataclass(frozen=True)
class GrpoConfig:
    enabled: bool = True
    run_policy: GrpoRunPolicy = "on_reward"
    threshold: float = 0.05
    gate_task_count: int = 4
    max_steps: int = 5
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 8
    rollout_attempts: int = 2
    vllm_server_base_url_env: str = "TRL_VLLM_SERVER_BASE_URL"


@dataclass(frozen=True)
class TrackingConfig:
    report_to: str = "wandb"
    project: str = "posttrainarena-benchflow"


@dataclass(frozen=True)
class PipelineConfig:
    source: Path
    model: str
    model_revision: str | None
    train_dataset: DatasetConfig
    eval_dataset: DatasetConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    sft: SftConfig = field(default_factory=SftConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output_root: Path = Path("runs")

    @property
    def sandbox(self) -> str:
        return self.runtime.sandbox or "daytona"

    def validate(self) -> None:
        errors: list[str] = []
        if self.harness.agent != "opencode":
            errors.append("harness.agent must be opencode")
        if self.harness.skill_mode not in {"no-skill", "with-skill"}:
            errors.append("harness.skill_mode must be no-skill or with-skill")
        if self.harness.usage_tracking != "required":
            errors.append("harness.usage_tracking must be required")
        if not _is_positive_int(self.harness.concurrency):
            errors.append("harness.concurrency must be positive")
        if not _is_positive_int(self.harness.sandbox_setup_timeout_sec):
            errors.append("harness.sandbox_setup_timeout_sec must be positive")
        if not _is_positive_int(self.harness.agent_idle_timeout_sec):
            errors.append("harness.agent_idle_timeout_sec must be positive")
        if not _is_positive_int(self.harness.agent_timeout_sec):
            errors.append("harness.agent_timeout_sec must be positive")
        if self.harness.reasoning_effort is not None and (
            not isinstance(self.harness.reasoning_effort, str)
            or not self.harness.reasoning_effort.strip()
        ):
            errors.append("harness.reasoning_effort must be a non-empty string")
        for label, value in (
            ("evaluation.base_model_env", self.evaluation.base_model_env),
            ("evaluation.student_model_env", self.evaluation.student_model_env),
            ("evaluation.base_url_env", self.evaluation.base_url_env),
            ("evaluation.api_key_env", self.evaluation.api_key_env),
        ):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} must be a non-empty string")
        if self.sandbox not in {"docker", "daytona"}:
            errors.append("runtime.sandbox must be docker or daytona")
        if self.runtime.num_generations < 2:
            errors.append("runtime.num_generations must be at least 2 for GRPO")
        if self.runtime.max_completion_length < 1:
            errors.append("runtime.max_completion_length must be positive")
        if not 0 <= self.grpo.threshold <= 1:
            errors.append("grpo.threshold must be between 0 and 1")
        if self.grpo.run_policy not in {"on_reward", "always"}:
            errors.append("grpo.run_policy must be on_reward or always")
        if self.grpo.gate_task_count < 1:
            errors.append("grpo.gate_task_count must be positive")
        if self.grpo.max_steps < 1:
            errors.append("grpo.max_steps must be positive")
        if not _is_positive_int(self.grpo.rollout_attempts):
            errors.append("grpo.rollout_attempts must be positive")
        if (
            not isinstance(self.grpo.vllm_server_base_url_env, str)
            or not self.grpo.vllm_server_base_url_env.strip()
        ):
            errors.append("grpo.vllm_server_base_url_env must be a non-empty string")
        if not _is_positive_int(self.teacher.max_attempts) or not _is_positive_int(
            self.teacher.min_verified
        ):
            errors.append("teacher max_attempts and min_verified must be positive")
        if (
            not isinstance(self.teacher.model, str)
            or not self.teacher.model.strip()
            or "/" not in self.teacher.model
        ):
            errors.append("teacher.model must use provider/model format")
        if (
            not isinstance(self.teacher.min_reward, int | float)
            or isinstance(self.teacher.min_reward, bool)
            or not 0 <= float(self.teacher.min_reward) <= 1
        ):
            errors.append("teacher.min_reward must be between 0 and 1")
        if not _is_positive_int(self.teacher.max_accepted_total_tokens):
            errors.append("teacher.max_accepted_total_tokens must be positive")
        if not _is_positive_int(self.teacher.max_accepted_tool_calls):
            errors.append("teacher.max_accepted_tool_calls must be positive")
        if self.sft.max_steps < 1:
            errors.append("sft.max_steps must be positive")
        for label, path in (
            ("train_dataset.task_list", self.train_dataset.task_list),
            ("eval_dataset.task_list", self.eval_dataset.task_list),
        ):
            if not path.is_file():
                errors.append(f"{label} does not exist: {path}")
        if self.sft.enabled and not self.teacher.enabled:
            errors.append("sft.enabled requires teacher.enabled")
        if errors:
            raise ValueError("Invalid pipeline config:\n- " + "\n- ".join(errors))


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path).expanduser().resolve()
    data = tomllib.loads(source.read_text())
    base = source.parent
    model = _table(data, "model")
    train = _table(data, "train_dataset")
    eval_data = _table(data, "eval_dataset")
    runtime = _table(data, "runtime")
    harness = _table(data, "harness", required=True)
    evaluation = _table(data, "evaluation", required=True)
    teacher = _table(data, "teacher")
    sft = _table(data, "sft")
    grpo = _table(data, "grpo")
    tracking = _table(data, "tracking")
    output = _table(data, "output")
    config = PipelineConfig(
        source=source,
        model=str(model["id"]),
        model_revision=str(model["revision"]) if model.get("revision") else None,
        train_dataset=DatasetConfig(
            repo_id=str(train["repo_id"]),
            revision=str(train["revision"]),
            task_list=_resolve(base, str(train["task_list"])),
            path=str(train.get("path", "tasks")),
        ),
        eval_dataset=DatasetConfig(
            repo_id=str(eval_data["repo_id"]),
            revision=str(eval_data["revision"]),
            task_list=_resolve(base, str(eval_data["task_list"])),
            path=str(eval_data.get("path", "tasks")),
        ),
        runtime=RuntimeConfig(**runtime),
        harness=HarnessConfig(**harness),
        evaluation=EvaluationConfig(**evaluation),
        teacher=TeacherConfig(**teacher),
        sft=SftConfig(**sft),
        grpo=GrpoConfig(**grpo),
        tracking=TrackingConfig(**tracking),
        output_root=_resolve(base, str(output.get("root", "../runs"))),
    )
    config.validate()
    return config
