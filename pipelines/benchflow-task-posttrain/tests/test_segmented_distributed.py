"""Two CPU ranks exercise segmented caching and DAPO token normalization."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from posttrainarena.benchflow_pipeline.segmented_trainer import (
    segmented_grpo_trainer_class,
)


ROOT = Path(__file__).resolve().parents[1]


def _segment(prompt_ids, completion_ids, initial_weight):
    return SimpleNamespace(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=[initial_weight * token for token in completion_ids],
    )


class TinyPolicy(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.forward_calls = 0

    def forward(self, token_ids):
        self.forward_calls += 1
        return self.weight * token_ids.float()


class DistributedAccelerator:
    num_processes = 2

    @staticmethod
    def gather(value):
        gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
        return torch.stack(gathered)

    @staticmethod
    def backward(loss):
        loss.backward()


class FakeBaseTrainer:
    loss_type = "dapo"
    importance_sampling_level = "token"
    use_vllm = True
    vllm_importance_sampling_correction = True
    vllm_importance_sampling_mode = "token_truncate"
    vllm_importance_sampling_clip_min = 0.01
    vllm_importance_sampling_clip_max = 100.0
    beta = 0.0

    def __init__(self, model, rollout, advantage):
        self.model = model
        self.rollout_func = SimpleNamespace(last_rollouts=[rollout])
        self._tokenizer = SimpleNamespace(pad_token_id=0)
        self.accelerator = DistributedAccelerator()
        self.advantage = advantage
        self.args = SimpleNamespace(
            delta=None, n_gpu=1, optim="adamw", torch_empty_cache_steps=None
        )
        self.current_gradient_accumulation_steps = 1
        self.model_accepts_loss_kwargs = False
        self.compute_loss_func = "non-None value to disable scaling"
        self._step = 0
        self.epsilon_low = self.epsilon_high = 0.2

    def _generate_and_score_completions(self, inputs):
        return {
            "prompt_ids": torch.zeros((1, 1), dtype=torch.long),
            "prompt_mask": torch.ones((1, 1), dtype=torch.long),
            "completion_ids": torch.zeros((1, 1), dtype=torch.long),
            "completion_mask": torch.ones((1, 1), dtype=torch.long),
            "advantages": torch.tensor([self.advantage]),
            "num_items_in_batch": torch.tensor(1),
        }

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size,
        compute_entropy=False,
    ):
        assert batch_size == input_ids.size(0)
        current = model(input_ids[:, -logits_to_keep:])
        return current, torch.zeros_like(current), None

    def _prepare_inputs(self, generation_batch):
        return generation_batch


def _worker() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    initial_weight = 0.2
    if rank == 0:
        segments = [_segment([7], [1, 2], initial_weight)]
        advantage = 1.0
    else:
        segments = [
            _segment([8], [3], initial_weight),
            _segment([9, 10], [4, 5], initial_weight),
        ]
        advantage = -0.5

    policy = TinyPolicy(initial_weight)
    ddp = DistributedDataParallel(policy)
    rollout = SimpleNamespace(tokens=SimpleNamespace(segments=segments))
    trainer = segmented_grpo_trainer_class(FakeBaseTrainer)(ddp, rollout, advantage)
    generated = trainer._generate_and_score_completions([{}])
    cached = trainer._segment_cache[0]["old_per_token_logps"].clone()
    prepared = trainer._prepare_inputs(generated)

    assert prepared["num_items_in_batch"].item() == 5
    assert prepared["completion_ids"][prepared["completion_mask"].bool()].numel() == (
        2 if rank == 0 else 3
    )
    with torch.no_grad():
        policy.weight.add_(0.05)
    assert torch.equal(trainer._segment_cache[0]["old_per_token_logps"], cached)

    trainer.training_step(ddp, generated)

    reference = TinyPolicy(initial_weight + 0.05)
    tokens = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    old = initial_weight * tokens
    advantages = torch.tensor([[1.0, 1.0, -0.5, -0.5, -0.5]])
    ratio = torch.exp(reference.weight * tokens - old)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    reference_loss = -torch.minimum(ratio * advantages, clipped * advantages).sum() / 5
    reference_loss.backward()

    assert torch.allclose(policy.weight.grad, reference.weight.grad, atol=1e-6)
    assert policy.forward_calls == 4
    assert trainer._step == 1
    dist.barrier()
    if rank == 0:
        print("SEGMENTED_DDP_OK", flush=True)
    dist.destroy_process_group()


def _launch() -> subprocess.CompletedProcess[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(port),
            __file__,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_unequal_segment_counts_match_single_process_dapo_gradient() -> None:
    completed = _launch()

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "SEGMENTED_DDP_OK" in completed.stdout


if __name__ == "__main__":
    _worker()
