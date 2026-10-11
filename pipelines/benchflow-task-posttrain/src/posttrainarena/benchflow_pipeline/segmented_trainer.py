from __future__ import annotations

from contextlib import nullcontext
import time
from typing import Any

import torch

from .segmented_rollout import pack_rollout_segments


def segmented_grpo_trainer_class(base_class: type) -> type:
    """Return a narrow GRPO trainer adapter for exact per-exchange segments."""

    class SegmentedGRPOTrainer(base_class):
        _segment_cache: dict[int, dict[str, Any]]

        def _validate_segmented_config(self) -> None:
            if self.loss_type != "dapo":
                raise RuntimeError("segmented rollouts support only DAPO loss")
            if self.importance_sampling_level != "token":
                raise RuntimeError(
                    "segmented rollouts require token importance sampling"
                )
            if not self.use_vllm or not self.vllm_importance_sampling_correction:
                raise RuntimeError(
                    "segmented rollouts require vLLM importance correction"
                )
            if self.vllm_importance_sampling_mode not in {
                "sequence_mask",
                "token_truncate",
            }:
                raise RuntimeError(
                    "segmented rollouts support only sequence_mask or token_truncate correction"
                )
            if self.beta != 0.0:
                raise RuntimeError(
                    "segmented rollouts do not support a reference KL term"
                )
            unsupported = (
                getattr(self, "use_liger_kernel", False)
                or getattr(self, "aux_loss_enabled", False)
                or getattr(self, "_entropy_bonus_enabled", False)
                or getattr(self, "top_entropy_quantile", 1.0) != 1.0
                or getattr(self, "off_policy_mask_threshold", None) is not None
            )
            if unsupported:
                raise RuntimeError(
                    "segmented chunking does not support auxiliary loss branches"
                )

        def _aligned_count(self, count: int, device: torch.device) -> int:
            counts = self.accelerator.gather(torch.tensor(count, device=device))
            return int(counts.max().item())

        def _row_batch(self, batch, index):
            if index >= len(batch["completion_ids"]):
                device = batch["completion_ids"].device
                token = torch.tensor([[self._tokenizer.pad_token_id]], device=device)
                zero = torch.zeros((1, 1), device=device)
                return {
                    "prompt_ids": token,
                    "prompt_mask": torch.ones_like(token),
                    "completion_ids": token,
                    "completion_mask": torch.zeros_like(token),
                    "old_per_token_logps": zero,
                    "sampling_per_token_logps": zero,
                    "importance_sampling_ratio": torch.ones_like(zero),
                    "advantages": zero[:, 0],
                    "num_items_in_batch": batch["num_items_in_batch"],
                }
            prompt_length = int(batch["prompt_mask"][index].sum().item())
            completion_length = int(batch["completion_mask"][index].sum().item())
            if not prompt_length or not completion_length:
                raise RuntimeError(
                    "exact segments require nonempty prompts and completions"
                )
            row = {}
            for key, value in batch.items():
                if key == "num_items_in_batch":
                    row[key] = value
                elif key in {"prompt_ids", "prompt_mask"}:
                    row[key] = value[index : index + 1, -prompt_length:]
                elif key == "advantages":
                    row[key] = value[index : index + 1]
                else:
                    row[key] = value[index : index + 1, :completion_length]
            return row

        @staticmethod
        def _segments(rollout: Any) -> Any:
            segments = getattr(getattr(rollout, "tokens", None), "segments", None)
            if segments is None:
                raise RuntimeError("collector rollout has no exact segments")
            return segments

        def _generate_and_score_completions(self, inputs):
            self._validate_segmented_config()
            output = super()._generate_and_score_completions(inputs)
            unsupported = {
                "pixel_values",
                "image_grid_thw",
                "pixel_attention_mask",
                "spatial_shapes",
                "num_tiles",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
                "image_position_ids",
                "num_images",
            }.intersection(output)
            if unsupported:
                raise RuntimeError(
                    "segmented rollouts support text-only batches; found "
                    + ", ".join(sorted(unsupported))
                )
            rollouts = list(getattr(self.rollout_func, "last_rollouts", ()))
            if len(rollouts) != len(output["advantages"]):
                raise RuntimeError(
                    "collector rollouts do not match normalized advantages"
                )

            groups = [self._segments(rollout) for rollout in rollouts]
            packed = pack_rollout_segments(
                groups,
                output["advantages"],
                pad_token_id=self._tokenizer.pad_token_id,
            )
            generation_rows = {
                "prompt_ids": packed.prompt_ids,
                "prompt_mask": packed.prompt_mask,
                "completion_ids": packed.completion_ids,
                "completion_mask": packed.completion_mask,
                "num_items_in_batch": torch.tensor(
                    0, device=packed.completion_ids.device
                ),
            }
            old_logps = torch.zeros_like(packed.sampling_per_token_logps)
            aligned_count = self._aligned_count(
                len(packed.completion_ids), packed.completion_ids.device
            )
            with torch.no_grad():
                for index in range(aligned_count):
                    row = self._row_batch(generation_rows, index)
                    logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        torch.cat([row["prompt_ids"], row["completion_ids"]], dim=1),
                        torch.cat([row["prompt_mask"], row["completion_mask"]], dim=1),
                        row["completion_ids"].size(1),
                        batch_size=1,
                    )
                    if index < len(packed.completion_ids):
                        old_logps[index, : logps.size(1)] = logps[0]

            logps_diff = (
                old_logps - packed.sampling_per_token_logps
            ) * packed.completion_mask
            if self.vllm_importance_sampling_mode == "token_truncate":
                ratio = torch.clamp(
                    torch.exp(logps_diff),
                    min=self.vllm_importance_sampling_clip_min,
                    max=self.vllm_importance_sampling_clip_max,
                )
            else:
                ratio = torch.zeros_like(logps_diff)
                for rollout_index in range(len(groups)):
                    rows = packed.rollout_indices == rollout_index
                    rollout_ratio = torch.exp(logps_diff[rows].sum())
                    invalid = (
                        self.vllm_importance_sampling_clip_min is not None
                        and rollout_ratio < self.vllm_importance_sampling_clip_min
                    ) or (
                        self.vllm_importance_sampling_clip_max is not None
                        and rollout_ratio > self.vllm_importance_sampling_clip_max
                    )
                    ratio[rows] = 0.0 if invalid else rollout_ratio
            local_tokens = packed.completion_mask.sum()
            global_tokens = self.accelerator.gather(local_tokens).sum()

            self._segment_cache = {}
            for rollout_index in range(len(groups)):
                rows = packed.rollout_indices == rollout_index
                self._segment_cache[rollout_index] = {
                    "prompt_ids": packed.prompt_ids[rows],
                    "prompt_mask": packed.prompt_mask[rows],
                    "completion_ids": packed.completion_ids[rows],
                    "completion_mask": packed.completion_mask[rows],
                    "sampling_per_token_logps": packed.sampling_per_token_logps[rows],
                    "old_per_token_logps": old_logps[rows],
                    "importance_sampling_ratio": ratio[rows],
                    "advantages": packed.advantages[rows],
                    "num_items_in_batch": global_tokens,
                }
            output["segment_rollout_ids"] = torch.arange(
                len(groups), device=output["advantages"].device
            )
            return output

        def _prepare_inputs(self, generation_batch):
            output = super()._prepare_inputs(generation_batch)
            ids = output.get("segment_rollout_ids")
            if ids is None:
                raise RuntimeError("prepared batch has no segment rollout IDs")
            selected = [self._segment_cache[int(index)] for index in ids.tolist()]
            expanded = {}
            for key in selected[0]:
                if key == "num_items_in_batch":
                    expanded[key] = selected[0][key]
                else:
                    expanded[key] = torch.cat([item[key] for item in selected], dim=0)
            return expanded

        def _chunk_loss(self, model, row):
            current, entropy, _ = self._get_per_token_logps_and_entropies(
                model,
                torch.cat([row["prompt_ids"], row["completion_ids"]], dim=1),
                torch.cat([row["prompt_mask"], row["completion_mask"]], dim=1),
                row["completion_ids"].size(1),
                batch_size=1,
                compute_entropy=True,
            )
            advantage = row["advantages"].unsqueeze(1)
            ratio = torch.exp(current - row["old_per_token_logps"])
            clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
            delta = getattr(self.args, "delta", None)
            bounded = torch.clamp(ratio, max=delta) if delta is not None else ratio
            per_token = -torch.minimum(bounded * advantage, clipped * advantage)
            per_token = per_token * row["importance_sampling_ratio"]
            mask = row["completion_mask"]
            loss = (per_token * mask).sum() / (
                row["num_items_in_batch"] / self.accelerator.num_processes
            )
            low = (ratio < 1 - self.epsilon_low) & (advantage < 0)
            high = (ratio > 1 + self.epsilon_high) & (advantage > 0)
            entropy_sum = (
                (entropy.detach() * mask).sum()
                if entropy is not None
                else loss.detach() * 0
            )
            stats = torch.stack(
                [
                    mask.sum(),
                    entropy_sum,
                    (low * mask).sum(),
                    (high * mask).sum(),
                    ((low | high) * mask).sum(),
                ]
            ).detach()
            return loss, stats

        def training_step(self, model, inputs, num_items_in_batch=None):
            """One outer Trainer step; aligned row graphs are released after each backward."""
            self._validate_segmented_config()
            distributed_type = str(getattr(self.accelerator, "distributed_type", ""))
            if any(name in distributed_type for name in ("DEEPSPEED", "MEGATRON")):
                raise RuntimeError(
                    "segmented training supports only DDP/FSDP or local execution"
                )
            if getattr(self.args, "n_gpu", 1) > 1:
                raise RuntimeError("segmented training does not support DataParallel")
            compute_loss_func = getattr(self, "compute_loss_func", None)
            if compute_loss_func not in (None, "non-None value to disable scaling"):
                raise RuntimeError(
                    "segmented training does not support custom loss functions"
                )
            if "lomo" in str(getattr(self.args, "optim", "")).lower():
                raise RuntimeError(
                    "segmented training does not support LOMO optimizers"
                )
            parallelism = getattr(self.accelerator, "parallelism_config", None)
            if parallelism is not None and getattr(parallelism, "cp_enabled", False):
                raise RuntimeError(
                    "segmented training does not support context parallelism"
                )
            start = time.perf_counter()
            model.train()
            optimizer = getattr(self, "optimizer", None)
            if callable(getattr(optimizer, "train", None)):
                optimizer.train()
            prepared = self._prepare_inputs(inputs)
            aligned_count = self._aligned_count(
                len(prepared["completion_ids"]), prepared["completion_ids"].device
            )
            total = torch.zeros((), device=prepared["completion_ids"].device)
            stats = torch.zeros(5, device=total.device)
            for index in range(aligned_count):
                row = self._row_batch(prepared, index)
                context = getattr(self, "compute_loss_context_manager", nullcontext)
                with context():
                    loss, row_stats = self._chunk_loss(model, row)
                if (
                    not getattr(self, "model_accepts_loss_kwargs", False)
                    or num_items_in_batch is None
                ) and compute_loss_func is None:
                    loss = loss / self.current_gradient_accumulation_steps
                self.accelerator.backward(loss)
                total += loss.detach()
                stats += row_stats
                del loss, row
            gathered = self.accelerator.gather(stats.unsqueeze(0)).reshape(-1, 5).sum(0)
            metrics = getattr(self, "_metrics", {}).get("train")
            if metrics is not None:
                count = gathered[0].clamp(min=1)
                for key, value in (
                    ("entropy", gathered[1] / count),
                    ("clip_ratio/low_mean", gathered[2] / count),
                    ("clip_ratio/high_mean", gathered[3] / count),
                    ("clip_ratio/region_mean", gathered[4] / count),
                ):
                    metrics[key].append(value.item())
            empty_steps = getattr(self.args, "torch_empty_cache_steps", None)
            if empty_steps is not None and self.state.global_step % empty_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._step += 1
            self._current_train_step_time = (
                getattr(self, "_current_train_step_time", 0.0)
                + time.perf_counter()
                - start
            )
            if (
                metrics is not None
                and self._step % self.current_gradient_accumulation_steps == 0
            ):
                metrics["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0
            return total

    SegmentedGRPOTrainer.__name__ = f"Segmented{base_class.__name__}"
    return SegmentedGRPOTrainer
