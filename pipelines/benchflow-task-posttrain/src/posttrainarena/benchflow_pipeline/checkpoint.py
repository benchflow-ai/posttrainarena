"""Portable adapter export and explicit CPU merge shared by SFT and GRPO.

A training worker leaves a PEFT adapter, its tokenizer, and provisional
``train_metrics.json`` in the adapter directory. This module turns that into
the merged checkpoint layout every downstream stage reads, in one process and
on the CPU so the merge never competes with trainer or vLLM GPU memory.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .io import directory_sha256, load_json, write_json


TRAIN_METRICS = "train_metrics.json"


def release_gpu_memory() -> None:
    """Return cached CUDA blocks once the caller has dropped its model references."""
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def export_merged_checkpoint(
    *,
    base_model: str,
    base_kwargs: dict[str, Any],
    adapter_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Merge ``adapter_dir`` into ``base_model`` on the CPU and publish metrics."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metrics = load_json(adapter_dir / TRAIN_METRICS)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="bfloat16",
        device_map="cpu",
        **base_kwargs,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    del base, merged
    release_gpu_memory()
    payload = {
        **metrics,
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(output_dir),
        "adapter_sha256": directory_sha256(adapter_dir),
        "merged_model_sha256": directory_sha256(output_dir),
    }
    write_json(output_dir / TRAIN_METRICS, payload)
    return payload
