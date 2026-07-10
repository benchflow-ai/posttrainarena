"""Tool-aware LoRA SFT and standalone checkpoint merge."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io import supported_kwargs, write_json


def render_rows(path: Path, tokenizer: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_num, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        tools = row.get("tool_defs", row.get("tools"))
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {line_num}: missing messages")
        rows.append(
            {
                "text": tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }
        )
    if not rows:
        raise ValueError(f"No SFT rows in {path}")
    return rows


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
    dataset = Dataset.from_list(render_rows(train_jsonl, tokenizer))
    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype="bfloat16", **model_kwargs
    )
    values = {
        "output_dir": str(adapter_dir),
        "max_steps": config.sft.max_steps,
        "learning_rate": config.sft.learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": config.sft.gradient_accumulation_steps,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [config.tracking.report_to]
        if config.tracking.report_to != "none"
        else "none",
        "run_name": f"{run_name}-sft",
        "dataset_text_field": "text",
        "max_length": config.sft.max_length,
        "packing": False,
    }
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**supported_kwargs(SFTConfig, values)),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=config.sft.lora_r,
            lora_alpha=config.sft.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    result = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
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
        "metrics": result.metrics,
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(output_dir),
    }
    write_json(output_dir / "train_metrics.json", metrics)
    return metrics
