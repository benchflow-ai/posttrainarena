"""Tool-aware LoRA SFT and standalone checkpoint merge."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io import directory_sha256, file_sha256, supported_kwargs, write_json


def load_trl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_num, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        completion = row.get("completion")
        tools = row.get("tools")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"row {line_num}: missing prompt messages")
        if (
            not isinstance(completion, list)
            or len(completion) != 1
            or not isinstance(completion[0], dict)
            or completion[0].get("role") != "assistant"
        ):
            raise ValueError(
                f"row {line_num}: completion must contain one assistant message"
            )
        if not isinstance(tools, list):
            raise ValueError(f"row {line_num}: missing tools")
        if "tool_defs" in row:
            raise ValueError(f"row {line_num}: TRL rows must use tools")
        rows.append(dict(row))
    if not rows:
        raise ValueError(f"No SFT rows in {path}")
    return rows


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(token, int) or isinstance(token, bool) for token in value)
    ):
        raise ValueError("Chat template did not return token IDs")
    return value


def build_tokenized_sft_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    tokenized = []
    max_prefix_mismatch = 0
    trained_tokens = 0
    for index, row in enumerate(rows):
        tools = row["tools"]
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                row["prompt"],
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                row["prompt"] + row["completion"],
                tools=tools,
                tokenize=True,
            )
        )
        common = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids, strict=False):
            if prompt_token != full_token:
                break
            common += 1
        mismatch = len(prompt_ids) - common
        max_prefix_mismatch = max(max_prefix_mismatch, mismatch)
        if common == 0 or mismatch > 8:
            raise ValueError(
                f"row {index}: prompt template diverges by {mismatch} tokens"
            )
        if len(full_ids) > max_length:
            raise ValueError(
                f"row {index}: tokenized length {len(full_ids)} exceeds {max_length}"
            )
        labels = [-100] * common + full_ids[common:]
        valid_tokens = sum(label != -100 for label in labels)
        if valid_tokens < 1:
            raise ValueError(f"row {index}: no trainable completion tokens")
        trained_tokens += valid_tokens
        tokenized.append(
            {
                "input_ids": full_ids,
                "labels": labels,
            }
        )
    return tokenized, {
        "max_prompt_prefix_mismatch": max_prefix_mismatch,
        "trained_tokens": trained_tokens,
    }


def train_sft(
    *,
    config: PipelineConfig,
    train_jsonl: Path,
    adapter_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if config.model_revision:
        model_kwargs["revision"] = config.model_revision
    tokenizer = AutoTokenizer.from_pretrained(config.model, **model_kwargs)
    if tokenizer is None:
        raise RuntimeError(f"Tokenizer failed to load for {config.model}")
    source_rows = load_trl_rows(train_jsonl)
    tokenized_rows, tokenization = build_tokenized_sft_rows(
        source_rows,
        tokenizer,
        max_length=config.sft.max_length,
    )
    dataset = Dataset.from_list(tokenized_rows)
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype="bfloat16", **model_kwargs
    )
    values = {
        "output_dir": str(adapter_dir),
        "learning_rate": config.sft.learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": config.sft.gradient_accumulation_steps,
        "gradient_checkpointing": config.sft.gradient_checkpointing,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [config.tracking.report_to]
        if config.tracking.report_to != "none"
        else "none",
        "run_name": f"{run_name}-sft",
        "max_length": config.sft.max_length,
        "packing": False,
        "completion_only_loss": False,
        "assistant_only_loss": False,
    }
    if config.sft.max_steps is None:
        values["num_train_epochs"] = config.sft.num_train_epochs
    else:
        values["max_steps"] = config.sft.max_steps
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**supported_kwargs(SFTConfig, values)),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=config.sft.lora_r,
            lora_alpha=config.sft.lora_alpha,
            lora_dropout=config.sft.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    result = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(
        adapter_dir / "adapter_dependency.json",
        {
            "schema_version": 1,
            "stage": "sft",
            "base_model": config.model,
            "base_revision": config.model_revision,
        },
    )
    del trainer, model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    base = AutoModelForCausalLM.from_pretrained(
        config.model, dtype="bfloat16", **model_kwargs
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    metrics = {
        "mode": "sft",
        "base_model": config.model,
        "model_revision": config.model_revision,
        "row_count": len(dataset),
        **tokenization,
        "num_train_epochs": (
            config.sft.num_train_epochs if config.sft.max_steps is None else None
        ),
        "max_steps": config.sft.max_steps,
        "quantization": None,
        "metrics": result.metrics,
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(output_dir),
        "train_jsonl_sha256": file_sha256(train_jsonl),
        "adapter_sha256": directory_sha256(adapter_dir),
        "merged_model_sha256": directory_sha256(output_dir),
    }
    write_json(output_dir / "train_metrics.json", metrics)
    return metrics
