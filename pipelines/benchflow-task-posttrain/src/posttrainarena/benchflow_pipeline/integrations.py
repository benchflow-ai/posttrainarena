"""Environment integration boundary shared by collection, evaluation, and RL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig


@dataclass(frozen=True)
class EnvironmentIntegration:
    """The exact environment-facing contract consumed by this pipeline."""

    train_dataset_rows: tuple[dict[str, Any], ...]
    environment_factory: Callable[[], Any]
    reward_funcs: tuple[Callable[..., list[float]], ...]
    _train_dataset_factory: Callable[[], Any]
    _close: Callable[[], None] = lambda: None

    @property
    def train_dataset(self) -> Any:
        return self._train_dataset_factory()

    def trainer_kwargs(self) -> dict[str, Any]:
        return {
            "train_dataset": self.train_dataset,
            "environment_factory": self.environment_factory,
            "reward_funcs": list(self.reward_funcs),
        }

    def close(self) -> None:
        self._close()


def build_environment_integration(
    *,
    config: PipelineConfig,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    reset_message: str,
) -> EnvironmentIntegration:
    """Build either transport around one canonical BenchFlow specification."""

    from benchflow.integrations.trl import BashHarnessConfig, BenchFlowSpec

    spec = BenchFlowSpec(
        tasks_dir=tasks_dir,
        include_tasks=task_ids,
        bash_harness=BashHarnessConfig(
            environment=config.sandbox,
            sandbox_user=config.runtime.sandbox_user,
            jobs_dir=jobs_dir,
            reset_message=reset_message,
            bash_timeout_sec=config.runtime.bash_timeout_sec,
            max_output_chars=config.runtime.max_output_chars,
        ),
    )
    if config.runtime.integration == "openenv":
        from .openenv import build_openenv_integration

        return build_openenv_integration(spec, config.runtime.openenv_url)
    return EnvironmentIntegration(
        train_dataset_rows=spec.train_dataset_rows,
        environment_factory=spec.environment_factory,
        reward_funcs=tuple(spec.reward_funcs),
        _train_dataset_factory=lambda: spec.train_dataset,
    )
