"""Training stages as owned worker processes, one GPU or many.

A stage's Accelerate profile is the single owner of its distributed type and
world size. Without a profile the stage runs in-process exactly as before.
With one, the pipeline launches ``sft-worker`` or ``grpo-worker`` as a child
job, waits for the whole group, and then merges the adapter in one process.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from .config import PipelineConfig
from .io import CommandRunner, file_sha256


WORKER_MODULE = "posttrainarena.benchflow_pipeline.cli"
SUPPORTED_DISTRIBUTED_TYPES = ("NO", "MULTI_GPU", "FSDP")
REQUIRED_KEYS = ("distributed_type", "num_processes", "mixed_precision")
DEPLOYMENT_KEYS = frozenset(
    {
        "gpu_ids",
        "machine_rank",
        "main_process_ip",
        "main_process_port",
        "rdzv_backend",
        "same_network",
    }
)
SEMANTIC_DEFAULTS = {
    "compute_environment": "LOCAL_MACHINE",
    "num_machines": 1,
    "use_cpu": False,
}
Stage = Literal["sft", "grpo"]
STAGES: tuple[Stage, ...] = ("sft", "grpo")


@dataclass(frozen=True)
class LaunchProfile:
    """An Accelerate profile, excluding device placement and rendezvous settings.

    ``settings`` holds the semantic keys with Accelerate's defaults filled in,
    so a profile that states a default and one that omits it are the same
    training identity.
    """

    path: Path
    settings: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> LaunchProfile:
        import yaml

        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Accelerate profile is not a mapping: {path}")
        stated = {
            key: value for key, value in raw.items() if key not in DEPLOYMENT_KEYS
        }
        profile = cls(path, {**SEMANTIC_DEFAULTS, **stated})
        profile.validate()
        return profile

    @classmethod
    def for_stage(cls, config: PipelineConfig, stage: Stage) -> LaunchProfile | None:
        path = getattr(config, stage).accelerate_config
        return None if path is None else cls.load(path)

    @property
    def distributed_type(self) -> str:
        return str(self.settings.get("distributed_type"))

    @property
    def num_processes(self) -> int:
        return int(self.settings.get("num_processes", 0))

    @property
    def distributed(self) -> bool:
        return self.distributed_type != "NO"

    @property
    def fingerprint(self) -> str:
        """Digest of the settings that change training, not deployment."""
        encoded = json.dumps(self.settings, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        errors = [
            f"{key} is required" for key in REQUIRED_KEYS if key not in self.settings
        ]
        if self.distributed_type not in SUPPORTED_DISTRIBUTED_TYPES:
            errors.append(
                f"distributed_type must be one of {SUPPORTED_DISTRIBUTED_TYPES}"
            )
        processes = self.settings.get("num_processes")
        if (
            not isinstance(processes, int)
            or isinstance(processes, bool)
            or processes < 1
        ):
            errors.append("num_processes must be a positive integer")
        elif (processes == 1) is self.distributed:
            errors.append(
                "distributed_type NO needs num_processes 1, and MULTI_GPU or FSDP "
                "need at least 2"
            )
        if self.settings["compute_environment"] != "LOCAL_MACHINE":
            errors.append("compute_environment must be LOCAL_MACHINE")
        if self.settings["num_machines"] != 1:
            errors.append("num_machines must be 1; multi-node launch is not supported")
        if self.settings.get("mixed_precision") != "bf16":
            errors.append("mixed_precision must be bf16 to match the recipe")
        if self.settings["use_cpu"]:
            errors.append("use_cpu must be false for GPU training")
        if self.settings.get("parallelism_config"):
            errors.append("parallelism_config is not supported by the training workers")
        fsdp = self.settings.get("fsdp_config") or {}
        if (
            self.distributed_type == "FSDP"
            and isinstance(fsdp, dict)
            and fsdp.get("fsdp_version") != 2
        ):
            errors.append("FSDP profiles must set fsdp_config.fsdp_version to 2")
        if self.distributed_type == "FSDP" and (
            not isinstance(fsdp, dict)
            or fsdp.get("fsdp_state_dict_type") != "FULL_STATE_DICT"
        ):
            errors.append(
                "FSDP profiles must set fsdp_config.fsdp_state_dict_type to "
                "FULL_STATE_DICT; Trainer.save_model writes no adapter otherwise"
            )
        if errors:
            raise ValueError(
                f"Invalid Accelerate profile {self.path}:\n- " + "\n- ".join(errors)
            )

    def summary(self) -> dict[str, Any]:
        """Training identity: what resume compares."""
        return {
            "distributed_type": self.distributed_type,
            "num_processes": self.num_processes,
            "fingerprint": self.fingerprint,
        }

    def manifest(self) -> dict[str, Any]:
        """Identity, the resolved settings, and provenance: what a launch record keeps."""
        return {
            **self.summary(),
            "settings": dict(self.settings),
            "profile": str(self.path),
            "profile_sha256": file_sha256(self.path),
        }

    def command(self, worker: str, arguments: list[str]) -> list[str]:
        """Argv for the worker: plain Python for one rank, Accelerate otherwise."""
        target = ["-m", WORKER_MODULE, worker, *arguments]
        if not self.distributed:
            return [sys.executable, *target]
        return [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(self.path),
            *target,
        ]


def stage_summary(config: PipelineConfig, stage: Stage) -> dict[str, Any] | None:
    """Semantic launch identity recorded in metrics and compared on resume."""
    profile = LaunchProfile.for_stage(config, stage)
    return None if profile is None else profile.summary()


def validate_grpo_topology(config: PipelineConfig, profile: LaunchProfile) -> None:
    """Mirror the TRL batch contract and the job-wide rollout budget."""
    ranks = profile.num_processes
    batch = config.generation_batch_size
    if batch % ranks:
        raise ValueError(
            f"grpo generation batch {batch} must be divisible by the "
            f"{ranks} trainer processes in {profile.path}"
        )
    if config.harness.concurrency < ranks:
        raise ValueError(
            f"harness.concurrency={config.harness.concurrency} leaves no rollout "
            f"slot for some of the {ranks} trainer processes in {profile.path}"
        )


def validate_profiles(config: PipelineConfig) -> dict[str, dict[str, Any] | None]:
    """Resolve both stage profiles, raising on any unusable one."""
    grpo_profile = LaunchProfile.for_stage(config, "grpo")
    if grpo_profile is not None:
        validate_grpo_topology(config, grpo_profile)
    return {stage: stage_summary(config, stage) for stage in STAGES}


def bundle_stage_profiles(
    data: dict[str, Any],
    *,
    source_dir: Path,
    output_dir: Path,
) -> None:
    """Copy referenced profiles next to a portable recipe and repoint it."""
    for stage in STAGES:
        table = data.get(stage) or {}
        reference = table.get("accelerate_config")
        if reference is None:
            continue
        source = Path(str(reference)).expanduser()
        if not source.is_absolute():
            source = (source_dir / source).resolve()
        destination = output_dir / "accelerate" / f"{stage}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        table["accelerate_config"] = f"accelerate/{destination.name}"


def verify_worker_topology(
    profile: LaunchProfile | None, *, timeout_sec: int = 1800
) -> dict[str, Any]:
    """Check the live process group against the profile before loading a model."""
    from accelerate import PartialState

    state = PartialState(timeout=timedelta(seconds=timeout_sec))
    expected = 1 if profile is None else profile.num_processes
    if state.num_processes != expected:
        raise RuntimeError(
            f"Worker started with {state.num_processes} processes; "
            f"the launch profile expects {expected}"
        )
    if expected == 1:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            raise RuntimeError(
                "A one-process trainer sees several GPUs; expose exactly one "
                "with CUDA_VISIBLE_DEVICES or launch a distributed profile"
            )
    return {
        "process_index": state.process_index,
        "num_processes": state.num_processes,
        "device": str(state.device),
    }


def _arguments(values: dict[str, Any]) -> list[str]:
    return [item for key, value in values.items() for item in (f"--{key}", str(value))]


def sft_worker_command(
    profile: LaunchProfile,
    *,
    config: PipelineConfig,
    train_jsonl: Path,
    adapter_dir: Path,
    run_name: str,
) -> list[str]:
    return profile.command(
        "sft-worker",
        _arguments(
            {
                "config": config.source,
                "train-jsonl": train_jsonl,
                "adapter-dir": adapter_dir,
                "run-name": run_name,
            }
        ),
    )


def grpo_worker_command(
    profile: LaunchProfile,
    *,
    config: PipelineConfig,
    model: str,
    tasks_dir: Path,
    task_ids_file: Path,
    jobs_dir: Path,
    adapter_dir: Path,
    run_name: str,
) -> list[str]:
    return profile.command(
        "grpo-worker",
        _arguments(
            {
                "config": config.source,
                "model": model,
                "tasks-dir": tasks_dir,
                "task-ids-file": task_ids_file,
                "jobs-dir": jobs_dir,
                "adapter-dir": adapter_dir,
                "run-name": run_name,
            }
        ),
    )


def run_sft_stage(
    *,
    config: PipelineConfig,
    runner: CommandRunner,
    train_jsonl: Path,
    adapter_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    profile = LaunchProfile.for_stage(config, "sft")
    if profile is None:
        from .sft import train_sft

        return train_sft(
            config=config,
            train_jsonl=train_jsonl,
            adapter_dir=adapter_dir,
            output_dir=output_dir,
            run_name=run_name,
        )
    from .checkpoint import export_merged_checkpoint
    from .sft import base_model_kwargs

    runner.run(
        "sft_worker",
        sft_worker_command(
            profile,
            config=config,
            train_jsonl=train_jsonl,
            adapter_dir=adapter_dir,
            run_name=run_name,
        ),
    )
    return export_merged_checkpoint(
        base_model=config.model,
        base_kwargs=base_model_kwargs(config),
        adapter_dir=adapter_dir,
        output_dir=output_dir,
    )


def run_grpo_stage(
    *,
    config: PipelineConfig,
    runner: CommandRunner,
    model: str,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    profile = LaunchProfile.for_stage(config, "grpo")
    if profile is None:
        from .grpo import train_grpo

        return train_grpo(
            config=config,
            model=model,
            tasks_dir=tasks_dir,
            task_ids=task_ids,
            jobs_dir=jobs_dir,
            adapter_dir=adapter_dir,
            output_dir=output_dir,
            run_name=run_name,
        )
    from .checkpoint import export_merged_checkpoint
    from .grpo import base_model_kwargs

    validate_grpo_topology(config, profile)
    task_ids_file = jobs_dir / "task_ids.txt"
    task_ids_file.parent.mkdir(parents=True, exist_ok=True)
    task_ids_file.write_text("\n".join(task_ids) + "\n")
    runner.run(
        "grpo_worker",
        grpo_worker_command(
            profile,
            config=config,
            model=model,
            tasks_dir=tasks_dir,
            task_ids_file=task_ids_file,
            jobs_dir=jobs_dir,
            adapter_dir=adapter_dir,
            run_name=run_name,
        ),
    )
    return export_merged_checkpoint(
        base_model=model,
        base_kwargs=base_model_kwargs(config, model),
        adapter_dir=adapter_dir,
        output_dir=output_dir,
    )
