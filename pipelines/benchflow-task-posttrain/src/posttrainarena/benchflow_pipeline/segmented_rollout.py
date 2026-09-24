from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor


class RolloutSegment(Protocol):
    prompt_ids: Sequence[int]
    completion_ids: Sequence[int]
    logprobs: Sequence[float]


@dataclass(frozen=True)
class SegmentedRolloutBatch:
    prompt_ids: Tensor
    prompt_mask: Tensor
    completion_ids: Tensor
    completion_mask: Tensor
    sampling_per_token_logps: Tensor
    advantages: Tensor
    rollout_indices: Tensor


def pack_rollout_segments(
    groups: Sequence[Sequence[RolloutSegment]],
    rollout_advantages: Tensor,
    *,
    pad_token_id: int,
) -> SegmentedRolloutBatch:
    """Flatten exact exchange segments while retaining their rollout advantage."""
    if rollout_advantages.ndim != 1 or len(rollout_advantages) != len(groups):
        raise ValueError("rollout advantages must be one-dimensional and match groups")
    if not groups or any(not group for group in groups):
        raise ValueError("rollout segment groups must be nonempty")

    segments = [segment for group in groups for segment in group]
    for segment in segments:
        if len(segment.completion_ids) != len(segment.logprobs):
            raise ValueError("segment completion IDs and logprobs must be aligned")

    max_prompt = max(len(segment.prompt_ids) for segment in segments)
    max_completion = max(len(segment.completion_ids) for segment in segments)
    count = len(segments)
    device = rollout_advantages.device
    prompt_ids = torch.full(
        (count, max_prompt), pad_token_id, dtype=torch.long, device=device
    )
    prompt_mask = torch.zeros_like(prompt_ids)
    completion_ids = torch.full(
        (count, max_completion), pad_token_id, dtype=torch.long, device=device
    )
    completion_mask = torch.zeros_like(completion_ids)
    sampling_logps = torch.zeros(
        (count, max_completion), dtype=torch.float32, device=device
    )

    rollout_indices: list[int] = []
    row = 0
    for rollout_index, group in enumerate(groups):
        for segment in group:
            prompt_length = len(segment.prompt_ids)
            completion_length = len(segment.completion_ids)
            if prompt_length:
                prompt_ids[row, -prompt_length:] = torch.as_tensor(
                    segment.prompt_ids, dtype=torch.long, device=device
                )
                prompt_mask[row, -prompt_length:] = 1
            if completion_length:
                completion_ids[row, :completion_length] = torch.as_tensor(
                    segment.completion_ids, dtype=torch.long, device=device
                )
                completion_mask[row, :completion_length] = 1
                sampling_logps[row, :completion_length] = torch.as_tensor(
                    segment.logprobs, dtype=torch.float32, device=device
                )
            rollout_indices.append(rollout_index)
            row += 1

    index_tensor = torch.tensor(rollout_indices, dtype=torch.long, device=device)
    return SegmentedRolloutBatch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        sampling_per_token_logps=sampling_logps,
        advantages=rollout_advantages[index_tensor],
        rollout_indices=index_tensor,
    )
