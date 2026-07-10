"""TRL-compatible tool environment backed by a real OpenEnv client."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..integrations import EnvironmentIntegration
from .client import PostTrainEnvClient
from .models import PostTrainAction, PostTrainObservation
from .server import LocalOpenEnvServer


class OpenEnvToolEnvironment:
    """Expose only the model tools while retaining protocol lifecycle control."""

    def __init__(self, base_url: str) -> None:
        self._client = PostTrainEnvClient(base_url=base_url).sync()
        self._client.connect()
        self.task_id: str | None = None
        self.reward = 0.0
        self.rollout_dir: Path | None = None
        self.last_returncode: int | None = None
        self._done = False
        self._closed = False

    def reset(self, **kwargs: Any) -> str:
        result = self._client.reset(**kwargs)
        self._update(result.observation)
        self._done = result.done
        return result.observation.output

    def run_bash(self, command: str) -> str:
        """Run a bash command in the BenchFlow task sandbox.

        Args:
            command: Bash command to execute in the task workspace.

        Returns:
            Combined standard output and standard error from the command.
        """
        return self._step(PostTrainAction(type="run_bash", command=command)).output

    def submit(self, answer: str) -> str:
        """Submit the final answer and run the BenchFlow verifier.

        Args:
            answer: Final answer to write into the task submission path.

        Returns:
            Confirmation text containing the verifier reward.
        """
        return self._step(PostTrainAction(type="submit", answer=answer)).output

    def _finalize(self) -> None:
        if not self._done:
            self._step(PostTrainAction(type="finalize"))

    def _close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def _state(self) -> Any:
        return self._client.state()

    def _step(self, action: PostTrainAction) -> PostTrainObservation:
        result = self._client.step(action)
        self._update(result.observation)
        self._done = result.done
        return result.observation

    def _update(self, observation: PostTrainObservation) -> None:
        self.task_id = observation.task_id
        self.reward = float(observation.reward or 0.0)
        self.rollout_dir = (
            Path(observation.rollout_dir) if observation.rollout_dir else None
        )
        self.last_returncode = observation.last_returncode


def openenv_environment_reward(
    completions: Sequence[Any],
    *,
    environments: Sequence[Any] | None = None,
    **_: Any,
) -> list[float]:
    if environments is None:
        return [0.0 for _ in completions]
    rewards: list[float] = []
    for index, _completion in enumerate(completions):
        environment = environments[index] if index < len(environments) else None
        if isinstance(environment, OpenEnvToolEnvironment):
            environment._finalize()
        value = getattr(environment, "reward", 0.0)
        rewards.append(
            float(value)
            if isinstance(value, int | float) and not isinstance(value, bool)
            else 0.0
        )
    return rewards


def build_openenv_integration(spec: Any, base_url: str | None) -> EnvironmentIntegration:
    local_server = None if base_url else LocalOpenEnvServer(spec.environment_factory)
    environments: list[OpenEnvToolEnvironment] = []

    def environment_factory() -> OpenEnvToolEnvironment:
        resolved_url = base_url or local_server.base_url
        environment = OpenEnvToolEnvironment(resolved_url)
        environments.append(environment)
        return environment

    def close() -> None:
        for environment in environments:
            environment._close()
        environments.clear()
        if local_server:
            local_server.close()

    return EnvironmentIntegration(
        train_dataset_rows=spec.train_dataset_rows,
        environment_factory=environment_factory,
        reward_funcs=(openenv_environment_reward,),
        _train_dataset_factory=lambda: spec.train_dataset,
        _close=close,
    )
