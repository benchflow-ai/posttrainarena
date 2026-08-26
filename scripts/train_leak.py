"""LoRA SFT for the leak control: train on the eval tasks' own oracle solutions.

This is contamination on purpose. Every row here is a task the evaluation will
score, so if the pipeline can detect improvement at all, the delta must come
back strongly positive. A flat or negative result means the instrument is
broken -- which would retroactively invalidate every negative we measured on
Fireworks, since none of them was ever checked against a known-positive control.

Heavy memorization settings deliberately: rank 32, lr 2e-4, many epochs over 11
rows. We are not trying to generalize.

    python3 train_leak.py --model /home/ubuntu/qwen36-27b \\
        --data sft_leak12.jsonl --out /home/ubuntu/leak-adapter --epochs 12
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=4096)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    # Render with the model's own chat template so training and serving agree;
    # a mismatch here shows up as a mysteriously flat delta.
    texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    ds = Dataset.from_dict({"text": texts})
    print(f"[leak] {len(rows)} rows, median chars {sorted(len(t) for t in texts)[len(texts)//2]}",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        max_seq_length=args.max_seq,
        dataset_text_field="text",
        report_to=["wandb"],
        run_name="leak-sft-qwen36-27b",
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[leak] adapter saved to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
