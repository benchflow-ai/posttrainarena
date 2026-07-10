"""Typed OpenEnv client for the PostTrain protocol."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import PostTrainAction, PostTrainObservation, PostTrainState


class PostTrainEnvClient(
    EnvClient[PostTrainAction, PostTrainObservation, PostTrainState]
):
    def _step_payload(self, action: PostTrainAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[PostTrainObservation]:
        observation_payload = dict(payload["observation"])
        observation_payload["reward"] = payload.get("reward")
        observation_payload["done"] = payload.get("done", False)
        observation = PostTrainObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> PostTrainState:
        return PostTrainState.model_validate(payload)
