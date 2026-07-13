"""TRL GRPO with OpenCode-driven BenchFlow rollouts."""

from __future__ import annotations

import json
import math
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Sequence

from .config import PipelineConfig
from .io import CommandRunner, supported_kwargs, write_json
from .opencode import evaluate


TASK_HANDLE_PREFIX = "benchflow-task://"


def _required_environment(name: str, *, label: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {label}: {name}")
    return value


def task_handle(task_id: str) -> str:
    if not task_id or "/" in task_id:
        raise ValueError(f"Invalid BenchFlow task ID: {task_id!r}")
    return f"{TASK_HANDLE_PREFIX}{task_id}"


def task_id_from_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str) or not prompt.startswith(TASK_HANDLE_PREFIX):
        raise ValueError(f"Unexpected GRPO prompt handle: {prompt!r}")
    return task_handle(prompt.removeprefix(TASK_HANDLE_PREFIX)).removeprefix(
        TASK_HANDLE_PREFIX
    )


def build_grpo_rows(tasks_dir: Path, task_ids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        if not task_dir.is_dir():
            raise FileNotFoundError(task_dir)
        rows.append(
            {
                "prompt": task_handle(task_id),
                "benchflow_task_id": task_id,
            }
        )
    return rows


def build_grpo_dataset(tasks_dir: Path, task_ids: list[str]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(build_grpo_rows(tasks_dir, task_ids))


def _model_init_kwargs(config: PipelineConfig, model: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
    }
    if model == config.model and config.model_revision:
        values["revision"] = config.model_revision
    return values


def _load_tokenizer(config: PipelineConfig, model: str | Path) -> Any:
    from transformers import AutoTokenizer

    values: dict[str, Any] = {
        "trust_remote_code": True,
        "fix_mistral_regex": True,
    }
    if str(model) == config.model and config.model_revision:
        values["revision"] = config.model_revision
    return AutoTokenizer.from_pretrained(str(model), **values)


def _close_weight_communicator(generation: Any) -> None:
    client = getattr(generation, "vllm_client", None)
    close = getattr(client, "close_communicator", None)
    if callable(close):
        close()


def sync_model_to_vllm(
    *,
    config: PipelineConfig,
    model: Any,
    accelerator: Any,
    processing_class: Any,
) -> None:
    from trl.generation.vllm_generation import VLLMGeneration

    server_base_url = _required_environment(
        config.grpo.vllm_server_base_url_env,
        label="TRL vLLM server endpoint",
    )
    synchronizer = VLLMGeneration(
        model=model,
        accelerator=accelerator,
        processing_class=processing_class,
        mode="server",
        server_base_url=server_base_url,
        max_completion_length=config.runtime.max_completion_length,
        logprobs=0,
        trust_remote_code=True,
    )
    try:
        synchronizer.sync_weights()
    finally:
        _close_weight_communicator(synchronizer)


def sync_reference_to_vllm(
    *,
    config: PipelineConfig,
    reference: str | Path,
) -> dict[str, Any]:
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM

    model_reference = str(reference)
    accelerator = Accelerator(mixed_precision="bf16")
    tokenizer = _load_tokenizer(config, model_reference)
    model = AutoModelForCausalLM.from_pretrained(
        model_reference,
        **_model_init_kwargs(config, model_reference),
    )
    model.to(accelerator.device)
    sync_model_to_vllm(
        config=config,
        model=model,
        accelerator=accelerator,
        processing_class=tokenizer,
    )
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return {
        "reference": model_reference,
        "student_model_env": config.evaluation.student_model_env,
        "vllm_server_base_url_env": config.grpo.vllm_server_base_url_env,
        "synced": True,
    }


def sync_checkpoint_to_vllm(
    *,
    config: PipelineConfig,
    checkpoint: Path,
) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    payload = sync_reference_to_vllm(
        config=config,
        reference=checkpoint,
    )
    return {
        **payload,
        "checkpoint": str(checkpoint),
    }


def verifier_reward(
    completions: Sequence[Any],
    *,
    rollout_reward: Sequence[float] | None = None,
    **_: Any,
) -> list[float]:
    if rollout_reward is None or len(rollout_reward) != len(completions):
        raise RuntimeError("OpenCode GRPO rollouts are missing verifier rewards")
    rewards = []
    for value in rollout_reward:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"Invalid OpenCode verifier reward: {value!r}")
        rewards.append(float(value))
    return rewards


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid OpenCode trajectory {path}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"OpenCode trajectory is empty: {path}")
    return rows


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise RuntimeError("Chat template did not return a non-empty token ID list")
    return value


def _chat_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    return _as_token_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _sampled_tokens(
    choice: dict[str, Any],
    tokenizer: Any,
) -> tuple[list[int], list[float]]:
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or not content:
        raise RuntimeError("OpenCode response has no sampled-token logprobs")

    token_ids = []
    sampled_logprobs = []
    explicit_ids = True
    raw_bytes = bytearray()
    for item in content:
        if not isinstance(item, dict):
            raise RuntimeError("OpenCode token logprob entry is not an object")
        value = item.get("logprob")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"Invalid sampled token logprob: {value!r}")
        sampled_logprobs.append(float(value))

        token_id = item.get("token_id")
        if isinstance(token_id, int) and not isinstance(token_id, bool):
            token_ids.append(token_id)
        else:
            explicit_ids = False

        byte_values = item.get("bytes")
        if isinstance(byte_values, list) and all(
            isinstance(byte, int) and not isinstance(byte, bool) and 0 <= byte <= 255
            for byte in byte_values
        ):
            raw_bytes.extend(byte_values)
        else:
            token = item.get("token")
            if not isinstance(token, str):
                raise RuntimeError("OpenCode token logprob has no token bytes")
            raw_bytes.extend(token.encode("utf-8"))

    if explicit_ids:
        return token_ids, sampled_logprobs

    try:
        text = bytes(raw_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("OpenCode token bytes are not valid UTF-8") from exc
    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_ids = _as_token_ids(encoded)
    if len(token_ids) != len(sampled_logprobs):
        raise RuntimeError(
            "OpenCode sampled-token count does not match the training tokenizer "
            f"({len(sampled_logprobs)} provider tokens vs {len(token_ids)} IDs)"
        )
    return token_ids, sampled_logprobs


def _exchange_parts(
    row: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    dict[str, Any],
    str | None,
]:
    request = row.get("request")
    response = row.get("response")
    request_body = request.get("body") if isinstance(request, dict) else None
    response_body = response.get("body") if isinstance(response, dict) else None
    if not isinstance(request_body, dict) or not isinstance(response_body, dict):
        raise RuntimeError("OpenCode trajectory exchange has no request/response body")
    messages = request_body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("OpenCode trajectory request has no messages")
    if any(not isinstance(message, dict) for message in messages):
        raise RuntimeError("OpenCode trajectory request contains invalid messages")
    tools = request_body.get("tools")
    if tools is not None and (
        not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise RuntimeError("OpenCode trajectory request contains invalid tools")
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("OpenCode trajectory response has no choice")
    completion_id = response_body.get("id")
    return (
        messages,
        tools,
        choices[0],
        completion_id if isinstance(completion_id, str) else None,
    )


@dataclass(frozen=True)
class RolloutTokens:
    prompt_ids: list[int]
    completion_ids: list[int]
    logprobs: list[float]
    env_mask: list[int]


def trajectory_to_rollout_tokens(
    path: Path,
    tokenizer: Any,
    *,
    max_completion_tokens: int,
    logprob_resolver: Callable[[str], dict[str, Any]] | None = None,
) -> RolloutTokens:
    rows = []
    for row in _load_jsonl(path):
        metadata = row.get("metadata")
        call_purpose = (
            metadata.get("call_purpose") if isinstance(metadata, dict) else None
        )
        if not isinstance(call_purpose, str):
            raise RuntimeError(
                "OpenCode trajectory exchange has no call-purpose metadata"
            )
        if call_purpose != "agent":
            continue
        response = row.get("response")
        status_code = (
            response.get("status_code") if isinstance(response, dict) else None
        )
        if not isinstance(status_code, int) or isinstance(status_code, bool):
            raise RuntimeError("OpenCode trajectory exchange has no response status")
        if 200 <= status_code < 300:
            rows.append(row)
    if not rows:
        raise RuntimeError("OpenCode trajectory has no agent exchanges")

    prompt_ids: list[int] | None = None
    completion_ids: list[int] = []
    sampled_logprobs: list[float] = []
    env_mask: list[int] = []
    current_ids: list[int] = []

    for index, row in enumerate(rows):
        messages, tools, choice, completion_id = _exchange_parts(row)
        request_ids = _chat_prompt_ids(tokenizer, messages, tools)
        if index == 0:
            prompt_ids = request_ids
            current_ids = list(request_ids)
        else:
            common = 0
            for left, right in zip(current_ids, request_ids, strict=False):
                if left != right:
                    break
                common += 1
            if common < len(prompt_ids):
                # OpenCode can refresh dynamic system context or compact history
                # between turns. Start a new causal segment at that request so
                # the returned prompt remains an exact model input.
                prompt_ids = request_ids
                completion_ids.clear()
                sampled_logprobs.clear()
                env_mask.clear()
            else:
                # OpenCode stores tool calls as structured messages. On the next
                # turn, the chat template may canonicalize a suffix of the prior
                # sampled tool-call text. Roll back that rewritten suffix, retain
                # sampled-token credit for the exact common prefix, and treat the
                # canonical replacement plus tool feedback as environment context.
                retained_completion = common - len(prompt_ids)
                del completion_ids[retained_completion:]
                del sampled_logprobs[retained_completion:]
                del env_mask[retained_completion:]
                external_ids = request_ids[common:]
                completion_ids.extend(external_ids)
                sampled_logprobs.extend([0.0] * len(external_ids))
                env_mask.extend([0] * len(external_ids))
            current_ids = list(request_ids)

        resolved_choice = choice
        if not isinstance(choice.get("logprobs"), dict):
            if logprob_resolver is None or completion_id is None:
                raise RuntimeError("OpenCode response has no sampled-token logprobs")
            resolved_choice = {
                **choice,
                "logprobs": logprob_resolver(completion_id),
            }
        generated_ids, generated_logprobs = _sampled_tokens(
            resolved_choice,
            tokenizer,
        )
        completion_ids.extend(generated_ids)
        sampled_logprobs.extend(generated_logprobs)
        env_mask.extend([1] * len(generated_ids))
        current_ids.extend(generated_ids)

    if prompt_ids is None or not completion_ids or not any(env_mask):
        raise RuntimeError("OpenCode trajectory produced no trainable model tokens")
    if len(completion_ids) > max_completion_tokens:
        raise RuntimeError(
            f"OpenCode trajectory has {len(completion_ids)} completion tokens; "
            f"limit is {max_completion_tokens}"
        )
    if not (len(completion_ids) == len(sampled_logprobs) == len(env_mask)):
        raise RuntimeError("OpenCode rollout token fields are not aligned")
    return RolloutTokens(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=sampled_logprobs,
        env_mask=env_mask,
    )


@dataclass(frozen=True)
class CollectedRollout:
    task_id: str
    reward: float
    rollout_dir: Path
    tokens: RolloutTokens


class OpenCodeRolloutCollector:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        model: str,
        tasks_dir: Path,
        jobs_dir: Path,
    ) -> None:
        self.config = config
        self.model = model
        self.tasks_dir = tasks_dir
        self.jobs_dir = jobs_dir
        self.records: list[dict[str, Any]] = []
        self._rollout_index = 0

    def _resolve_bridge_logprobs(self, completion_id: str) -> dict[str, Any]:
        import httpx

        base_url = os.environ.get(self.config.evaluation.control_url_env)
        if not base_url:
            base_url = _required_environment(
                self.config.evaluation.base_url_env,
                label="OpenCode model bridge endpoint",
            )
        base_url = base_url.rstrip("/")
        api_key = _required_environment(
            self.config.evaluation.api_key_env,
            label="OpenCode model bridge API key",
        )
        response = httpx.get(
            f"{base_url}/benchflow/logprobs/{completion_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        logprobs = payload.get("logprobs") if isinstance(payload, dict) else None
        if not isinstance(logprobs, dict):
            raise RuntimeError(f"Model bridge returned no logprobs for {completion_id}")
        return logprobs

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        if not prompts:
            raise ValueError("OpenCode GRPO rollout batch must not be empty")
        tokenizer = trainer.processing_class
        start_index = self._rollout_index
        self._rollout_index += len(prompts)
        requests = [
            (start_index + offset, task_id_from_prompt(prompt))
            for offset, prompt in enumerate(prompts)
        ]

        def collect(request: tuple[int, str]) -> CollectedRollout:
            rollout_index, task_id = request
            return self._collect_one(
                rollout_index=rollout_index,
                task_id=task_id,
                tokenizer=tokenizer,
                trainer=trainer,
            )

        with ThreadPoolExecutor(
            max_workers=min(self.config.harness.concurrency, len(requests))
        ) as executor:
            collected = list(executor.map(collect, requests))
        return {
            "prompt_ids": [rollout.tokens.prompt_ids for rollout in collected],
            "completion_ids": [rollout.tokens.completion_ids for rollout in collected],
            "logprobs": [rollout.tokens.logprobs for rollout in collected],
            "env_mask": [rollout.tokens.env_mask for rollout in collected],
            "rollout_reward": [rollout.reward for rollout in collected],
            "benchflow_task_id": [rollout.task_id for rollout in collected],
            "rollout_dir": [str(rollout.rollout_dir) for rollout in collected],
        }

    def _collect_one(
        self,
        *,
        rollout_index: int,
        task_id: str,
        tokenizer: Any,
        trainer: Any,
    ) -> CollectedRollout:
        global_step = int(getattr(trainer.state, "global_step", 0))
        rank = int(getattr(trainer.accelerator, "process_index", 0))
        rollout_root = (
            self.jobs_dir
            / f"step-{global_step:06d}"
            / f"rank-{rank:02d}"
            / f"rollout-{rollout_index:06d}"
        )
        failures = []
        for attempt in range(1, self.config.grpo.rollout_attempts + 1):
            attempt_root = rollout_root / f"attempt-{attempt:02d}"
            try:
                metrics_path = attempt_root / "metrics.json"
                payload = evaluate(
                    config=self.config,
                    runner=CommandRunner(cwd=self.config.source.parent),
                    stage=f"grpo_rollout_{global_step:06d}_{rollout_index:06d}",
                    model=self.model,
                    model_role="student",
                    tasks_dir=self.tasks_dir,
                    task_ids=[task_id],
                    jobs_dir=attempt_root / "jobs",
                    metrics_path=metrics_path,
                    capture_token_logprobs=True,
                )
                rollout = self._materialize_rollout(
                    payload=payload,
                    attempt_root=attempt_root,
                    attempt=attempt,
                    task_id=task_id,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    rank=rank,
                )
                (attempt_root / "rollout_error.json").unlink(missing_ok=True)
                return rollout
            except Exception as exc:
                failure = {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }
                failures.append(failure)
                write_json(attempt_root / "rollout_error.json", failure)
        raise RuntimeError(
            f"OpenCode GRPO rollout failed for {task_id} after "
            f"{self.config.grpo.rollout_attempts} attempts: {failures}"
        )

    def _materialize_rollout(
        self,
        *,
        payload: dict[str, Any],
        attempt_root: Path,
        attempt: int,
        task_id: str,
        tokenizer: Any,
        global_step: int,
        rank: int,
    ) -> CollectedRollout:
        health = payload.get("health")
        rows = health.get("rows") if isinstance(health, dict) else None
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
        ):
            raise RuntimeError("OpenCode GRPO health summary has no rollout")
        health_row = rows[0]
        reward = health_row.get("reward")
        if (
            not isinstance(reward, int | float)
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise RuntimeError(f"Invalid verifier reward: {reward!r}")
        rollout_dir = Path(str(health_row.get("rollout_dir") or ""))
        trajectory_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
        tokens = trajectory_to_rollout_tokens(
            trajectory_path,
            tokenizer,
            max_completion_tokens=self.config.runtime.max_completion_length,
            logprob_resolver=self._resolve_bridge_logprobs,
        )
        record = {
            "task_id": task_id,
            "reward": float(reward),
            "rollout_dir": str(rollout_dir),
            "attempt": attempt,
            "global_step": global_step,
            "rank": rank,
            "prompt_tokens": len(tokens.prompt_ids),
            "completion_tokens": len(tokens.completion_ids),
            "action_tokens": sum(tokens.env_mask),
        }
        write_json(attempt_root / "rollout.json", record)
        write_json(
            attempt_root / "grpo_tokens.json",
            {
                "prompt_ids": tokens.prompt_ids,
                "completion_ids": tokens.completion_ids,
                "logprobs": tokens.logprobs,
                "action_mask": tokens.env_mask,
                "reward": float(reward),
            },
        )
        self.records.append(record)
        return CollectedRollout(
            task_id=task_id,
            reward=float(reward),
            rollout_dir=rollout_dir,
            tokens=tokens,
        )


def train_grpo(
    *,
    config: PipelineConfig,
    model: str,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    dataset = build_grpo_dataset(tasks_dir, task_ids)
    collector = OpenCodeRolloutCollector(
        config=config,
        model=model,
        tasks_dir=tasks_dir,
        jobs_dir=jobs_dir,
    )
    vllm_server_base_url = _required_environment(
        config.grpo.vllm_server_base_url_env,
        label="TRL vLLM server endpoint",
    )
    processing_class = _load_tokenizer(config, model)
    generation_batch_size = (
        config.grpo.generation_batch_size
        or config.runtime.num_generations * config.harness.concurrency
    )
    values = {
        "output_dir": str(adapter_dir),
        "run_name": run_name,
        "bf16": True,
        "report_to": (
            [config.tracking.report_to]
            if config.tracking.report_to != "none"
            else "none"
        ),
        "remove_unused_columns": False,
        "max_completion_length": config.runtime.max_completion_length,
        "log_completions": config.grpo.log_completions,
        "use_vllm": True,
        "vllm_mode": "server",
        "vllm_server_base_url": vllm_server_base_url,
        "vllm_importance_sampling_correction": True,
        "logging_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": config.grpo.gradient_accumulation_steps,
        "gradient_checkpointing": config.grpo.gradient_checkpointing,
        "generation_batch_size": generation_batch_size,
        "learning_rate": config.grpo.learning_rate,
        "save_strategy": "no",
        "num_generations": config.runtime.num_generations,
        "model_init_kwargs": _model_init_kwargs(config, model),
    }
    if config.grpo.max_steps is None:
        values["num_train_epochs"] = config.grpo.num_train_epochs
    else:
        values["max_steps"] = config.grpo.max_steps
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**supported_kwargs(GRPOConfig, values)),
        train_dataset=dataset,
        reward_funcs=[verifier_reward],
        rollout_func=collector,
        processing_class=processing_class,
        peft_config=LoraConfig(
            r=config.grpo.lora_r,
            lora_alpha=config.grpo.lora_alpha,
            lora_dropout=config.grpo.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    try:
        result = trainer.train()
    finally:
        _close_weight_communicator(getattr(trainer, "vllm_generation", None))
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None:
        processing_class.save_pretrained(str(adapter_dir))
    write_json(
        adapter_dir / "adapter_dependency.json",
        {
            "schema_version": 1,
            "stage": "grpo",
            "base_checkpoint": model,
            "published_base_sibling": "../sft-merged",
            "original_base_model": config.model,
            "original_base_revision": config.model_revision,
        },
    )
    del trainer
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    base = AutoModelForCausalLM.from_pretrained(
        model,
        **_model_init_kwargs(config, model),
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    if processing_class is not None:
        processing_class.save_pretrained(str(output_dir))
    payload = {
        "mode": "grpo",
        "harness": config.harness.agent,
        "model": model,
        "task_ids": task_ids,
        "num_train_epochs": (
            config.grpo.num_train_epochs if config.grpo.max_steps is None else None
        ),
        "max_steps": config.grpo.max_steps,
        "generation_batch_size": generation_batch_size,
        "quantization": None,
        "metrics": result.metrics,
        "jobs_dir": str(jobs_dir),
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(output_dir),
        "resume_policy": "restart-stage",
        "rollout_count": len(collector.records),
        "rollouts": collector.records,
        "rollout_contract": {
            "token_ids": "training-tokenizer-aligned",
            "logprobs": "provider-sampled",
            "action_mask": "model-tokens-only",
            "reward": "benchflow-verifier",
            "endpoint_sync": "trl-vllm-sync-weights",
            "vllm_server_base_url_env": config.grpo.vllm_server_base_url_env,
        },
    }
    write_json(output_dir / "train_metrics.json", payload)
    return payload
