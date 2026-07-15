"""Typed TOML configuration for the public BenchFlow pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Literal


BENCHFLOW_COMMIT = "2a97db55947d6742b765ad34ddd91d74c20d625f"
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


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(float(value))
        and float(value) > 0
    )


def _tuple_field(
    table: dict[str, Any],
    name: str,
    default: tuple[str, ...],
) -> tuple[Any, ...]:
    value = table.get(name, default)
    if not isinstance(value, list | tuple):
        raise ValueError(f"harness.{name} must be a TOML array")
    return tuple(value)


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
    num_generations: int = 8


@dataclass(frozen=True)
class HarnessConfig:
    agent: str = "opencode"
    skill_mode: HarnessSkillMode = "no-skill"
    usage_tracking: UsageTrackingPolicy = "required"
    external_directory_allow: tuple[str, ...] = ("/home/user/input/**",)
    deny_bash_patterns: tuple[str, ...] = ("*<<*",)
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
    control_url_env: str = "BENCHFLOW_MODEL_BRIDGE_CONTROL_URL"
    api_key_env: str = "BENCHFLOW_PROVIDER_API_KEY"
    sync_base_to_vllm: bool = False


@dataclass(frozen=True)
class TeacherConfig:
    enabled: bool = True
    model: str = "glm/glm-5.1"
    source_model: str | None = None
    source_revision: str | None = None
    max_attempts: int = 3
    min_verified: int = 1
    require_all_tasks: bool = True
    min_reward: float = 1.0
    max_accepted_total_tokens: int = 200000
    max_accepted_tool_calls: int = 50


@dataclass(frozen=True)
class SftConfig:
    enabled: bool = True
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    learning_rate: float = 2e-5
    max_length: int = 4096
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass(frozen=True)
class GrpoConfig:
    enabled: bool = True
    run_policy: GrpoRunPolicy = "on_reward"
    threshold: float = 0.05
    gate_task_count: int = 4
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    log_completions: bool = False
    generation_batch_size: int | None = None
    rollout_attempts: int = 2
    require_reward_variance: bool = False
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
        if (
            not isinstance(self.harness.external_directory_allow, tuple)
            or not self.harness.external_directory_allow
            or any(
                not isinstance(path, str) or not path.startswith("/")
                for path in self.harness.external_directory_allow
            )
        ):
            errors.append(
                "harness.external_directory_allow must contain absolute paths"
            )
        if (
            not isinstance(self.harness.deny_bash_patterns, tuple)
            or not self.harness.deny_bash_patterns
            or any(
                not isinstance(pattern, str) or not pattern
                for pattern in self.harness.deny_bash_patterns
            )
        ):
            errors.append("harness.deny_bash_patterns must contain strings")
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
            ("evaluation.control_url_env", self.evaluation.control_url_env),
            ("evaluation.api_key_env", self.evaluation.api_key_env),
        ):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} must be a non-empty string")
        if not isinstance(self.evaluation.sync_base_to_vllm, bool):
            errors.append("evaluation.sync_base_to_vllm must be boolean")
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
        if self.grpo.max_steps is not None and not _is_positive_int(
            self.grpo.max_steps
        ):
            errors.append("grpo.max_steps must be positive")
        if not _is_positive_number(self.grpo.num_train_epochs):
            errors.append("grpo.num_train_epochs must be positive")
        if not _is_positive_int(self.grpo.gradient_accumulation_steps):
            errors.append("grpo.gradient_accumulation_steps must be positive")
        if not isinstance(self.grpo.gradient_checkpointing, bool):
            errors.append("grpo.gradient_checkpointing must be boolean")
        if not _is_positive_int(self.grpo.lora_r):
            errors.append("grpo.lora_r must be positive")
        if not _is_positive_int(self.grpo.lora_alpha):
            errors.append("grpo.lora_alpha must be positive")
        if (
            not isinstance(self.grpo.lora_dropout, int | float)
            or isinstance(self.grpo.lora_dropout, bool)
            or not isfinite(float(self.grpo.lora_dropout))
            or not 0 <= float(self.grpo.lora_dropout) < 1
        ):
            errors.append("grpo.lora_dropout must be between 0 and 1")
        if not isinstance(self.grpo.log_completions, bool):
            errors.append("grpo.log_completions must be boolean")
        if self.grpo.generation_batch_size is not None and (
            not _is_positive_int(self.grpo.generation_batch_size)
            or self.grpo.generation_batch_size % self.runtime.num_generations != 0
        ):
            errors.append(
                "grpo.generation_batch_size must be positive and divisible by "
                "runtime.num_generations"
            )
        if not _is_positive_int(self.grpo.rollout_attempts):
            errors.append("grpo.rollout_attempts must be positive")
        if not isinstance(self.grpo.require_reward_variance, bool):
            errors.append("grpo.require_reward_variance must be boolean")
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
        if self.teacher.source_model is not None:
            if (
                not isinstance(self.teacher.source_model, str)
                or "/" not in self.teacher.source_model
            ):
                errors.append("teacher.source_model must use org/model format")
            elif self.teacher.model.rsplit("/", 1)[-1].lower() != (
                self.teacher.source_model.rsplit("/", 1)[-1].lower()
            ):
                errors.append("teacher.model must match teacher.source_model")
            if (
                not isinstance(self.teacher.source_revision, str)
                or len(self.teacher.source_revision) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in self.teacher.source_revision.lower()
                )
            ):
                errors.append("teacher.source_revision must be a 40-character SHA")
        elif self.teacher.source_revision is not None:
            errors.append("teacher.source_revision requires teacher.source_model")
        if not isinstance(self.teacher.require_all_tasks, bool):
            errors.append("teacher.require_all_tasks must be boolean")
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
        if self.sft.max_steps is not None and not _is_positive_int(self.sft.max_steps):
            errors.append("sft.max_steps must be positive")
        if not _is_positive_number(self.sft.num_train_epochs):
            errors.append("sft.num_train_epochs must be positive")
        if not _is_positive_int(self.sft.gradient_accumulation_steps):
            errors.append("sft.gradient_accumulation_steps must be positive")
        if not isinstance(self.sft.gradient_checkpointing, bool):
            errors.append("sft.gradient_checkpointing must be boolean")
        if not _is_positive_int(self.sft.lora_r):
            errors.append("sft.lora_r must be positive")
        if not _is_positive_int(self.sft.lora_alpha):
            errors.append("sft.lora_alpha must be positive")
        if (
            not isinstance(self.sft.lora_dropout, int | float)
            or isinstance(self.sft.lora_dropout, bool)
            or not isfinite(float(self.sft.lora_dropout))
            or not 0 <= float(self.sft.lora_dropout) < 1
        ):
            errors.append("sft.lora_dropout must be between 0 and 1")
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
        harness=HarnessConfig(
            **{
                **harness,
                "external_directory_allow": _tuple_field(
                    harness,
                    "external_directory_allow",
                    HarnessConfig().external_directory_allow,
                ),
                "deny_bash_patterns": _tuple_field(
                    harness,
                    "deny_bash_patterns",
                    HarnessConfig().deny_bash_patterns,
                ),
            }
        ),
        evaluation=EvaluationConfig(**evaluation),
        teacher=TeacherConfig(**teacher),
        sft=SftConfig(**sft),
        grpo=GrpoConfig(**grpo),
        tracking=TrackingConfig(**tracking),
        output_root=_resolve(base, str(output.get("root", "../runs"))),
    )
    config.validate()
    return config
