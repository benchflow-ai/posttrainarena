"""Which training tasks GRPO samples, and what happened to each of them.

Two samplers choose the tasks of each GRPO generation batch:

- ``trl`` (default, grpo-v1 behavior): TRL's ``RepeatSampler``. Each epoch it draws one
  ``torch.randperm`` of the task rows (seeded by ``grpo.seed``, TRL's ``args.seed``), cuts it into
  chunks of ``prompts_per_step`` tasks and drops the last partial chunk. With a fixed
  ``max_steps`` smaller than one epoch, most of a collection is never sampled.
- ``cover``: a seeded, balanced walk through the whole collection. Python's
  ``random.Random(seed)`` shuffles the task list; the shuffled lists are concatenated and cut
  into steps of ``prompts_per_step`` tasks. Within a step a task is not repeated while the
  collection has at least ``prompts_per_step`` tasks (a repeat at a shuffle boundary is deferred
  to the next step). Every task is sampled within the first ceil(N / prompts_per_step) steps and,
  at any step, exposure counts differ by at most one. The full schedule is fixed before training,
  written to ``reports/train_sampler.json``, and fed to TRL in order (``shuffle_dataset=False``).

One generation batch holds ``prompts_per_step`` groups of ``num_generations`` rollouts. With
``gradient_accumulation_steps`` equal to the generation batch size (required by ``cover``), one
optimizer step consumes exactly one generation batch, so "step" means both.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from typing import Any

from .config import PipelineConfig

PASS_THRESHOLD = 1.0

TRL_SAMPLER_DESCRIPTION = (
    "TRL RepeatSampler: one torch.randperm of the task rows per epoch (seeded by grpo.seed), "
    "cut into chunks of prompts_per_step tasks; the last partial chunk of each epoch is dropped."
)
COVER_SAMPLER_DESCRIPTION = (
    "Balanced cover: random.Random(grpo.seed) shuffles of the task list, concatenated and cut "
    "into steps of prompts_per_step tasks, no repeat within a step when the collection allows."
)


def effective_generation_batch_size(config: PipelineConfig) -> int:
    return (
        config.grpo.generation_batch_size
        or config.runtime.num_generations * config.harness.concurrency
    )


def prompts_per_step(config: PipelineConfig) -> int:
    """Distinct task groups in one generation batch."""
    return effective_generation_batch_size(config) // config.runtime.num_generations


def cover_schedule(
    task_ids: Sequence[str],
    *,
    prompts_per_step: int,
    steps: int,
    seed: int,
) -> list[list[str]]:
    """Task IDs of every step, balanced over the whole collection (see module docstring)."""
    unique = list(dict.fromkeys(task_ids))
    if not unique:
        raise ValueError("cover_schedule needs at least one task")
    if prompts_per_step < 1 or steps < 1:
        raise ValueError("prompts_per_step and steps must be positive")
    rng = random.Random(seed)
    distinct = prompts_per_step <= len(unique)
    pending: deque[str] = deque()
    schedule: list[list[str]] = []
    for _ in range(steps):
        chunk: list[str] = []
        deferred: list[str] = []
        while len(chunk) < prompts_per_step:
            if not pending:
                order = list(unique)
                rng.shuffle(order)
                pending.extend(order)
            task_id = pending.popleft()
            if distinct and task_id in chunk:
                deferred.append(task_id)
                continue
            chunk.append(task_id)
        pending.extendleft(reversed(deferred))
        schedule.append(chunk)
    return schedule


def _exposure_summary(
    task_ids: Sequence[str], steps: Sequence[Sequence[str]]
) -> dict[str, Any]:
    unique = list(dict.fromkeys(task_ids))
    counts = Counter(task_id for step in steps for task_id in step)
    seen: set[str] = set()
    full_coverage_step = None
    for index, step in enumerate(steps, start=1):
        seen.update(step)
        if full_coverage_step is None and len(seen & set(unique)) == len(unique):
            full_coverage_step = index
    return {
        "covered_task_count": sum(1 for task_id in unique if counts[task_id]),
        "uncovered_task_ids": [task_id for task_id in unique if not counts[task_id]],
        "min_exposures": min((counts[task_id] for task_id in unique), default=0),
        "max_exposures": max((counts[task_id] for task_id in unique), default=0),
        "steps_to_full_coverage": full_coverage_step,
    }


def sampler_plan(config: PipelineConfig, task_ids: Sequence[str]) -> dict[str, Any]:
    """The sampler contract of a GRPO run, including the full schedule for ``cover``."""
    unique = list(dict.fromkeys(task_ids))
    per_step = prompts_per_step(config)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "sampler": config.grpo.task_sampler,
        "seed": config.grpo.seed,
        "collection_size": len(unique),
        "task_ids": unique,
        "prompts_per_step": per_step,
        "num_generations": config.runtime.num_generations,
        "generation_batch_size": effective_generation_batch_size(config),
        "gradient_accumulation_steps": config.grpo.gradient_accumulation_steps,
        "max_steps": config.grpo.max_steps,
        "num_train_epochs": (
            config.grpo.num_train_epochs if config.grpo.max_steps is None else None
        ),
        "rollout_failure_policy": config.grpo.rollout_failure_policy,
    }
    if config.grpo.task_sampler == "cover":
        if config.grpo.max_steps is None:
            raise ValueError("grpo.task_sampler = cover requires grpo.max_steps")
        schedule = cover_schedule(
            unique,
            prompts_per_step=per_step,
            steps=config.grpo.max_steps,
            seed=config.grpo.seed,
        )
        plan.update(
            description=COVER_SAMPLER_DESCRIPTION,
            planned_steps=schedule,
            planned_exposures=_exposure_summary(unique, schedule),
        )
    else:
        plan.update(
            description=TRL_SAMPLER_DESCRIPTION,
            planned_steps=None,
            planned_exposures=None,
        )
    return plan


def coverage_problem(config: PipelineConfig, task_ids: Sequence[str]) -> str | None:
    """Why a ``cover`` recipe cannot sample every training task, or None."""
    if (
        not config.grpo.enabled
        or config.grpo.task_sampler != "cover"
        or not config.grpo.require_full_coverage
        or config.grpo.max_steps is None
    ):
        return None
    unique = list(dict.fromkeys(task_ids))
    capacity = config.grpo.max_steps * prompts_per_step(config)
    if capacity >= len(unique):
        return None
    return (
        f"grpo.max_steps ({config.grpo.max_steps}) x prompts per step "
        f"({prompts_per_step(config)}) = {capacity} task groups cannot cover the "
        f"{len(unique)} training tasks; raise max_steps or generation_batch_size, or set "
        "grpo.require_full_coverage = false"
    )


def _population_variance(values: Sequence[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def train_task_stats(
    task_ids: Sequence[str],
    *,
    records: Sequence[Mapping[str, Any]],
    masked_records: Sequence[Mapping[str, Any]] = (),
    failed_attempts: Sequence[Mapping[str, Any]] = (),
    num_generations: int,
) -> dict[str, Any]:
    """Per-task GRPO accounting for every task in the collection, sampled or not.

    ``rollouts`` counts every rollout the sampler assigned to the task. ``scored_rollouts`` had a
    verifier reward and entered the loss; ``infra_errors`` and ``trajectory_errors`` were masked
    (no loss, excluded from the group baseline), never scored 0. ``failed_attempts`` counts
    retried attempts, including ones that later succeeded. ``reward_variance`` is the population
    variance of the scored rewards (p(1-p) for pass/fail rewards). A group has signal when at
    least two scored rollouts differ in reward.
    """
    unique = list(dict.fromkeys(task_ids))
    for record in (*records, *masked_records):
        task_id = record.get("task_id")
        if isinstance(task_id, str) and task_id not in unique:
            unique.append(task_id)
    tasks: dict[str, dict[str, Any]] = {
        task_id: {
            "steps": set(),
            "groups": set(),
            "rewards": [],
            "infra_errors": 0,
            "trajectory_errors": 0,
            "truncated_rollouts": 0,
            "failed_attempts": 0,
        }
        for task_id in unique
    }
    groups: dict[Any, dict[str, Any]] = {}
    for record in records:
        entry = tasks[record["task_id"]]
        entry["steps"].add(record.get("global_step"))
        entry["groups"].add(record.get("group_index"))
        entry["rewards"].append(float(record["reward"]))
        entry["truncated_rollouts"] += int(bool(record.get("truncated")))
        group = groups.setdefault(
            record.get("group_index"),
            {"task_id": record["task_id"], "rewards": [], "masked": 0},
        )
        group["rewards"].append(float(record["reward"]))
    for record in masked_records:
        entry = tasks[record["task_id"]]
        entry["steps"].add(record.get("global_step"))
        entry["groups"].add(record.get("group_index"))
        kind = "trajectory_errors" if record.get("kind") == "trajectory" else "infra_errors"
        entry[kind] += 1
        group = groups.setdefault(
            record.get("group_index"),
            {"task_id": record["task_id"], "rewards": [], "masked": 0},
        )
        group["masked"] += 1
    for attempt in failed_attempts:
        task_id = attempt.get("task_id")
        if task_id in tasks:
            tasks[task_id]["failed_attempts"] += 1
    group_counts: dict[str, Counter[str]] = {task_id: Counter() for task_id in unique}
    for group in groups.values():
        rewards = group["rewards"]
        counter = group_counts[group["task_id"]]
        if len(rewards) >= 2 and max(rewards) - min(rewards) > 1e-12:
            counter["with_variance"] += 1
        elif rewards and min(rewards) >= PASS_THRESHOLD:
            counter["all_pass"] += 1
        elif rewards and max(rewards) <= 0.0:
            counter["all_fail"] += 1
        if not rewards:
            counter["fully_masked"] += 1
    rows = {}
    for task_id in unique:
        entry = tasks[task_id]
        rewards = entry["rewards"]
        passes = sum(1 for reward in rewards if reward >= PASS_THRESHOLD)
        masked = entry["infra_errors"] + entry["trajectory_errors"]
        rows[task_id] = {
            "sampled_steps": sorted(
                step for step in entry["steps"] if isinstance(step, int)
            ),
            "groups": len(entry["groups"]),
            "rollouts": len(rewards) + masked,
            "scored_rollouts": len(rewards),
            "passes": passes,
            "pass_rate": passes / len(rewards) if rewards else None,
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "reward_variance": _population_variance(rewards),
            "infra_errors": entry["infra_errors"],
            "trajectory_errors": entry["trajectory_errors"],
            "truncated_rollouts": entry["truncated_rollouts"],
            "failed_attempts": entry["failed_attempts"],
            "groups_with_variance": group_counts[task_id]["with_variance"],
            "all_pass_groups": group_counts[task_id]["all_pass"],
            "all_fail_groups": group_counts[task_id]["all_fail"],
            "fully_masked_groups": group_counts[task_id]["fully_masked"],
        }
    totals = {
        key: sum(row[key] for row in rows.values())
        for key in (
            "groups",
            "rollouts",
            "scored_rollouts",
            "passes",
            "infra_errors",
            "trajectory_errors",
            "truncated_rollouts",
            "failed_attempts",
            "groups_with_variance",
            "all_pass_groups",
            "all_fail_groups",
            "fully_masked_groups",
        )
    }
    masked_total = totals["infra_errors"] + totals["trajectory_errors"]
    totals["masked_rollout_fraction"] = (
        masked_total / totals["rollouts"] if totals["rollouts"] else None
    )
    return {
        "schema_version": 1,
        "num_generations": num_generations,
        "pass_threshold": PASS_THRESHOLD,
        "task_count": len(unique),
        "sampled_task_count": sum(1 for row in rows.values() if row["rollouts"]),
        "unsampled_task_ids": [
            task_id for task_id, row in rows.items() if not row["rollouts"]
        ],
        "totals": totals,
        "tasks": rows,
    }


def sampler_report(
    plan: Mapping[str, Any],
    *,
    batches: Sequence[Mapping[str, Any]],
    trainer_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The sampler plan plus the task IDs actually sampled in every generation batch."""
    steps = [
        {
            key: batch.get(key)
            for key in (
                "generation_index",
                "global_step",
                "task_ids",
                "rollouts",
                "masked_rollouts",
            )
        }
        for batch in batches
    ]
    planned = plan.get("planned_steps")
    mismatched = []
    if planned is not None:
        for index, step in enumerate(steps):
            expected = planned[index] if index < len(planned) else None
            if expected is None or Counter(expected) != Counter(
                batch_task_multiset(batches[index])
            ):
                mismatched.append(index)
    actual = [batch_task_multiset(batch) for batch in batches]
    return {
        **plan,
        "trainer": dict(trainer_args or {}),
        "steps": steps,
        "actual_exposures": _exposure_summary(plan.get("task_ids") or [], actual),
        "plan_mismatch_steps": mismatched if planned is not None else None,
    }


def batch_task_multiset(batch: Mapping[str, Any]) -> list[str]:
    """Task ID of every group in a generation batch (a task can own several groups)."""
    groups = batch.get("groups")
    if isinstance(groups, list) and groups:
        return [group["task_id"] for group in groups]
    return list(batch.get("task_ids") or [])
