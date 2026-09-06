"""Rank helpers that degrade to single-process semantics without Accelerate.

Every helper reads the trainer's ``accelerator`` when one exists and otherwise
behaves as a lone process, so the same training code runs in-process, in a
one-process worker, and under ``accelerate launch``.
"""

from __future__ import annotations

import os
from typing import Any


BOOTSTRAP_VARIABLES = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    }
)
BOOTSTRAP_PREFIX = "TORCHELASTIC_"


def _accelerator(trainer: Any) -> Any:
    return getattr(trainer, "accelerator", None)


def process_index(trainer: Any) -> int:
    return int(getattr(_accelerator(trainer), "process_index", 0))


def num_processes(trainer: Any) -> int:
    return int(getattr(_accelerator(trainer), "num_processes", 1))


def is_main_process(trainer: Any) -> bool:
    return bool(getattr(_accelerator(trainer), "is_main_process", True))


def wait_for_everyone(trainer: Any) -> None:
    barrier = getattr(_accelerator(trainer), "wait_for_everyone", None)
    if callable(barrier):
        barrier()


def gather_records(trainer: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect every rank's records in rank order; a collective on all ranks."""
    if num_processes(trainer) == 1:
        return list(records)
    from accelerate.utils import gather_object

    return list(gather_object(records))


def gather_peak_gpu_memory_mib(trainer: Any) -> list[int]:
    """Every rank's peak allocated CUDA memory so far, in rank order; a collective."""
    import torch

    peak = torch.cuda.max_memory_allocated() // 2**20 if torch.cuda.is_available() else 0
    rows = gather_records(trainer, [{"peak_mib": int(peak)}])
    return [row["peak_mib"] for row in rows]


def rollout_quota(total: int, world_size: int, rank: int) -> int:
    """Split a job-wide concurrency budget across ranks, lower ranks first."""
    if total < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid rollout quota request: total={total}, "
            f"world_size={world_size}, rank={rank}"
        )
    share, remainder = divmod(total, world_size)
    return share + int(rank < remainder)


def subprocess_environment() -> dict[str, str]:
    """This process's environment without its process-group bootstrap.

    Rollout subprocesses must not inherit the trainer's rank, world size, or
    rendezvous address, or a distributed-aware tool inside them would join the
    trainer's group instead of running as an ordinary program.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if key not in BOOTSTRAP_VARIABLES and not key.startswith(BOOTSTRAP_PREFIX)
    }


def close_process_group() -> None:
    """Dispose of the live process group; a no-op outside one."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def declare_fsdp_layer_classes(model: Any) -> list[str] | None:
    """Name the model's transformer layer classes for Accelerate's FSDP wrapping.

    Transformers lists every class a model family may contain, so a text-only
    Qwen3.5 causal LM still names its vision block, and Accelerate refuses a
    class it cannot find. Under an FSDP launch, and unless the profile named
    the classes itself, declare only the no-split classes present in this
    model. Returns the declared names, or ``None`` when nothing was declared.
    """
    if os.environ.get("ACCELERATE_USE_FSDP", "").lower() != "true":
        return None
    if os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP"):
        return None
    present = {type(module).__name__ for module in model.modules()}
    classes = [name for name in getattr(model, "_no_split_modules", None) or [] if name in present]
    if not classes:
        return None
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = ",".join(classes)
    return classes
