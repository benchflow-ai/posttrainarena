from types import SimpleNamespace

import pytest
import torch

from posttrainarena.benchflow_pipeline.segmented_rollout import pack_rollout_segments


def _segment(prompt_ids, completion_ids, logprobs):
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
    )


def test_pack_rollout_segments_preserves_unequal_groups() -> None:
    groups = [
        [_segment([1, 2], [10, 11], [-0.1, -0.2])],
        [
            _segment([3], [12], [-0.3]),
            _segment([4, 5, 6], [13, 14, 15], [-0.4, -0.5, -0.6]),
        ],
    ]

    batch = pack_rollout_segments(groups, torch.tensor([1.25, -0.75]), pad_token_id=99)

    assert batch.prompt_ids.tolist() == [[99, 1, 2], [99, 99, 3], [4, 5, 6]]
    assert batch.prompt_mask.tolist() == [[0, 1, 1], [0, 0, 1], [1, 1, 1]]
    assert batch.completion_ids.tolist() == [
        [10, 11, 99],
        [12, 99, 99],
        [13, 14, 15],
    ]
    assert batch.completion_mask.tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
    assert batch.sampling_per_token_logps == pytest.approx(
        torch.tensor([[-0.1, -0.2, 0.0], [-0.3, 0.0, 0.0], [-0.4, -0.5, -0.6]])
    )
    assert batch.advantages.tolist() == [1.25, -0.75, -0.75]
    assert batch.rollout_indices.tolist() == [0, 1, 1]
    assert batch.completion_ids[batch.completion_mask.bool()].tolist() == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]


def test_dapo_segment_gradient_matches_global_token_mean() -> None:
    groups = [
        [_segment([1], [10, 11], [-0.1, -0.2])],
        [_segment([2], [12], [-0.3]), _segment([3], [13, 14, 15], [-0.4, -0.5, -0.6])],
    ]
    batch = pack_rollout_segments(groups, torch.tensor([1.25, -0.75]), pad_token_id=0)
    current_logps = batch.sampling_per_token_logps.clone().requires_grad_()
    ratios = (current_logps - batch.sampling_per_token_logps).exp()
    token_losses = -ratios * batch.advantages[:, None] * batch.completion_mask
    global_tokens = batch.completion_mask.sum()
    # Two ranks own one rollout each; DDP averages their gradients.
    rank_losses = [
        token_losses[batch.rollout_indices == rank].sum() / (global_tokens / 2)
        for rank in range(2)
    ]
    loss = torch.stack(rank_losses).mean()
    loss.backward()
    assert loss.item() == pytest.approx(-(2 * 1.25 - 4 * 0.75) / 6)
    expected = -batch.advantages[:, None] * batch.completion_mask / 6
    torch.testing.assert_close(current_logps.grad, expected)


def test_pack_rollout_segments_rejects_misaligned_completion_logprobs() -> None:
    groups = [[_segment([1], [10, 11], [-0.1])]]

    with pytest.raises(ValueError, match="completion IDs and logprobs"):
        pack_rollout_segments(groups, torch.tensor([1.0]), pad_token_id=0)


@pytest.mark.parametrize("groups", [[], [[]]])
def test_pack_rollout_segments_rejects_empty_groups(groups) -> None:
    advantages = torch.empty(len(groups))

    with pytest.raises(ValueError, match="groups must be nonempty"):
        pack_rollout_segments(groups, advantages, pad_token_id=0)
