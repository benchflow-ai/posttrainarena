"""Two real ranks over Gloo: gather order, global rollout identity, failure exit.

Run by pytest through ``torch.distributed.run`` with this file as the script.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAIL_RANK = "POSTTRAINARENA_TEST_FAIL_RANK"
TIMEOUT_STAGE = "POSTTRAINARENA_TEST_TIMEOUT_STAGE"


def _timeout_worker() -> None:
    import argparse
    from dataclasses import replace
    from functools import partial
    import time
    from unittest.mock import patch

    from accelerate import PartialState
    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group
    from transformers import TrainingArguments

    from posttrainarena.benchflow_pipeline import grpo, sft
    from posttrainarena.benchflow_pipeline.cli import _run_worker
    from posttrainarena.benchflow_pipeline.config import load_config

    stage = os.environ[TIMEOUT_STAGE]
    timeout = int(os.environ["POSTTRAINARENA_TEST_TIMEOUT_SECONDS"])
    delay = int(os.environ["POSTTRAINARENA_TEST_DELAY_SECONDS"])
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        **{
            stage: replace(
                getattr(config, stage),
                ddp_timeout=timeout,
                accelerate_config=ROOT / "configs/accelerate/ddp-2gpu.yaml",
            )
        },
    )

    def inspect_training_group(**kwargs):
        def effective_timeout():
            group = _get_default_group()._get_backend(torch.device("cpu"))
            return group.options._timeout.total_seconds()

        assert effective_timeout() == timeout
        args = TrainingArguments(output_dir="unused", use_cpu=True, ddp_timeout=timeout)
        assert args.device.type == "cpu"
        assert effective_timeout() == timeout
        dist.barrier()
        if dist.get_rank() == 1:
            time.sleep(delay)
        value = torch.ones(1)
        dist.all_reduce(value)
        assert value.item() == 2
        print(f"TIMEOUT_OK stage={stage} seconds={timeout}", flush=True)
        return {}

    # Keep the real CLI and both process-group initializers; no model is needed.
    module = sft if stage == "sft" else grpo
    setattr(module, f"train_{stage}_adapter", inspect_training_group)
    # Select real Gloo groups for this CPU-only test of the GPU worker entry.
    with patch("accelerate.PartialState", partial(PartialState, cpu=True)):
        _run_worker(
            argparse.Namespace(
                command=f"{stage}-worker",
                train_jsonl=ROOT / "unused.jsonl",
                model="unused",
                tasks_dir=ROOT,
                task_ids_file=ROOT / "task-lists/data-agent-train-15.txt",
                jobs_dir=ROOT,
                adapter_dir=ROOT,
                run_name="group-timeout",
            ),
            config,
        )


def _worker() -> None:
    if TIMEOUT_STAGE in os.environ:
        _timeout_worker()
        return
    from types import SimpleNamespace

    from accelerate import Accelerator

    from posttrainarena.benchflow_pipeline.config import load_config
    from posttrainarena.benchflow_pipeline.distributed import (
        gather_records,
        wait_for_everyone,
    )
    from posttrainarena.benchflow_pipeline.grpo import OpenCodeRolloutCollector

    accelerator = Accelerator(cpu=True)
    trainer = SimpleNamespace(accelerator=accelerator)
    if os.environ.get(FAIL_RANK) == str(accelerator.process_index):
        raise RuntimeError(f"rank {accelerator.process_index} failed on purpose")
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    collector = OpenCodeRolloutCollector(
        config=config, model="x", tasks_dir=ROOT, jobs_dir=ROOT
    )
    slots = [slot for _ in range(2) for slot in collector._slots(3, trainer)]
    local = [
        {"rank": slot.rank, "index": slot.index, "call": slot.generation_call}
        for slot in slots
    ]

    gathered = gather_records(trainer, local)
    wait_for_everyone(trainer)

    if accelerator.is_main_process:
        assert [row["rank"] for row in gathered] == [0] * 6 + [1] * 6, gathered
        assert sorted(row["index"] for row in gathered) == list(range(12)), gathered
        assert [row["index"] for row in gathered[:6]] == [0, 1, 2, 6, 7, 8]
        assert [row["call"] for row in gathered] == [0, 0, 0, 1, 1, 1] * 2
        print("GATHER_OK")


def _launch(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
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
        timeout=600,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **env},
    )


def test_two_ranks_gather_in_rank_order_with_global_indexes() -> None:
    completed = _launch({})

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "GATHER_OK" in completed.stdout


def test_a_failing_rank_fails_the_launch() -> None:
    completed = _launch({FAIL_RANK: "1"})

    assert completed.returncode != 0
    assert "failed on purpose" in completed.stderr


@pytest.mark.parametrize("stage", ["sft", "grpo"])
def test_worker_applies_timeout_before_trainer_initialization(stage: str) -> None:
    completed = _launch(
        {
            TIMEOUT_STAGE: stage,
            "POSTTRAINARENA_TEST_TIMEOUT_SECONDS": "30",
            "POSTTRAINARENA_TEST_DELAY_SECONDS": "1",
            "ACCELERATE_USE_CPU": "true",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert f"TIMEOUT_OK stage={stage} seconds=30" in completed.stdout


def test_worker_collective_enforces_configured_timeout() -> None:
    completed = _launch(
        {
            TIMEOUT_STAGE: "grpo",
            "POSTTRAINARENA_TEST_TIMEOUT_SECONDS": "5",
            "POSTTRAINARENA_TEST_DELAY_SECONDS": "8",
            "ACCELERATE_USE_CPU": "true",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    assert completed.returncode != 0
    assert "timed out" in completed.stderr.lower()


if __name__ == "__main__":
    _worker()
