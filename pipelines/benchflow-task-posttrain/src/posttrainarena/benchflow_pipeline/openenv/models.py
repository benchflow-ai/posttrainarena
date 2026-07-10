"""Wire models for the PostTrain OpenEnv protocol."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State


class PostTrainAction(Action):
    type: Literal["run_bash", "submit", "finalize"]
    command: str | None = None
    answer: str | None = None


class PostTrainObservation(Observation):
    output: str = ""
    task_id: str | None = None
    rollout_dir: str | None = None
    last_returncode: int | None = None


class PostTrainState(State):
    task_id: str | None = None
    done: bool = False
    reward: float = 0.0
    rollout_dir: str | None = None
