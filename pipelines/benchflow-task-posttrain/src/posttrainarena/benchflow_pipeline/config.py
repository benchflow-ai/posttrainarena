"""Typed TOML configuration for the public BenchFlow pipeline."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


BENCHFLOW_COMMIT = "6eaa14344bd835a3c2c5c31a31470ef994b24a80"
GrpoRunPolicy = Literal["on_reward", "always"]


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    revision: str
    task_list: Path
    path: str = "tasks"


@dataclass(frozen=True)
class RuntimeConfig:
    integration: str = "benchflow"
    environment: str | None = None
    sandbox: str | None = None
    openenv_url: str | None = None
    sandbox_user: str | None = "agent"
    bash_timeout_sec: int = 120
    max_output_chars: int = 8192
    max_completion_length: int = 2048
    max_tool_calling_iterations: int = 25
    num_generations: int = 2
    use_vllm: bool = False


@dataclass(frozen=True)
class TeacherConfig:
    enabled: bool = True
    model: str = "glm-5.1"
    api_key_env: str = "GLM_API_KEY"
    base_url_env: str = "GLM_BASE_URL"
    max_attempts: int = 3
    max_tokens: int = 4096
    min_verified: int = 1
    temperature: float = 0.2


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
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    sft: SftConfig = field(default_factory=SftConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output_root: Path = Path("runs")

    @property
    def sandbox(self) -> str:
        return self.runtime.sandbox or self.runtime.environment or "daytona"

    def validate(self) -> None:
        errors: list[str] = []
        if self.runtime.integration not in {"benchflow", "openenv"}:
            errors.append("runtime.integration must be benchflow or openenv")
        if self.runtime.openenv_url and self.runtime.integration != "openenv":
            errors.append("runtime.openenv_url requires integration = openenv")
        if (
            self.runtime.sandbox is not None
            and self.runtime.environment is not None
            and self.runtime.sandbox != self.runtime.environment
        ):
            errors.append("runtime.sandbox conflicts with legacy runtime.environment")
        if self.sandbox not in {"docker", "daytona"}:
            errors.append("runtime.sandbox must be docker or daytona")
        if self.runtime.num_generations < 2:
            errors.append("runtime.num_generations must be at least 2 for GRPO")
        if self.runtime.max_completion_length < 1:
            errors.append("runtime.max_completion_length must be positive")
        if self.runtime.max_tool_calling_iterations < 1:
            errors.append("runtime.max_tool_calling_iterations must be positive")
        if not 0 <= self.grpo.threshold <= 1:
            errors.append("grpo.threshold must be between 0 and 1")
        if self.grpo.run_policy not in {"on_reward", "always"}:
            errors.append("grpo.run_policy must be on_reward or always")
        if self.grpo.gate_task_count < 1:
            errors.append("grpo.gate_task_count must be positive")
        if self.grpo.max_steps < 1:
            errors.append("grpo.max_steps must be positive")
        if self.teacher.max_attempts < 1 or self.teacher.min_verified < 1:
            errors.append("teacher max_attempts and min_verified must be positive")
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
        teacher=TeacherConfig(**teacher),
        sft=SftConfig(**sft),
        grpo=GrpoConfig(**grpo),
        tracking=TrackingConfig(**tracking),
        output_root=_resolve(base, str(output.get("root", "../runs"))),
    )
    config.validate()
    return config
