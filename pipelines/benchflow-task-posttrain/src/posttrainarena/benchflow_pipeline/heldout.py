"""Held-out pass rates, paired deltas and standard errors over suites and trials.

Notation, for one held-out suite s with tasks i = 1..n and trials k:

- y_ik ∈ {0, 1} is pass@1 of task i in trial k (verifier reward >= PASS_THRESHOLD, the
  BenchFlow definition of "passed"). A (task, trial) cell with no healthy scored rollout is an
  infrastructure error: it is excluded, not scored 0, and counted.
- ybar_i = mean of y_ik over the K_i trials that scored. A task with K_i = 0 has no estimate.
- Pass rate p = mean of ybar_i over the tasks that scored at least once.
- Paired effect: d_i = ybar_i(final) - ybar_i(baseline) over the tasks scored in both arms,
  Δ = mean of d_i = p(final) - p(baseline) on that paired task set.

Standard error. Tasks are treated as a sample from the population of tasks the suite stands for,
and trials as independent draws within each task (two-stage sampling). Then

    SE(Δ) = sqrt( s_d^2 / n ),   s_d^2 = sum_i (d_i - Δ)^2 / (n - 1)

and E[s_d^2] = σ²_task + mean_i( σ²_base,i / K_base,i + σ²_final,i / K_final,i ): one sample
variance of the per-task differences carries both the between-task spread of the true effect
and the within-task trial noise, shrunk by the number of trials. This is the task-clustered,
paired standard error of Miller, "Adding Error Bars to Evals" (arXiv 2411.00640), sections 2-4.
When every paired task has at least two scored trials in both arms, the trial part is also
estimated on its own, w = mean_i( v_base,i / K_base,i + v_final,i / K_final,i ) with v the
unbiased within-task variance across trials, and reported as `variance_components.trial`,
`variance_components.task = max(0, s_d^2 - w)` and `stderr_fixed_tasks = sqrt(w / n)` (the
standard error if the task set is regarded as fixed and only trials are resampled).

Pooled over suites, suites are strata: Δ_pooled = sum_s n_s Δ_s / N (every paired task weighs
the same) and SE_pooled = sqrt( sum_s (n_s / N)^2 SE_s^2 ). Pass rates pool the same way.
`ci95` is the normal approximation Δ ± 1.96 SE.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .opencode import is_scored_row

PASS_THRESHOLD = 1.0
Z95 = 1.959963984540054

STDERR_METHOD = (
    "Per suite: SE(delta) = sqrt(s_d^2 / n), s_d^2 the sample variance over paired tasks of "
    "d_i = mean over trials of pass@1 after minus before (task-clustered, paired; includes "
    "task and trial variation). Pooled: suites as strata, SE = sqrt(sum_s (n_s/N)^2 SE_s^2). "
    "Infra-error cells are excluded, not scored 0. ci95 = delta +/- 1.96 SE."
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _ci95(estimate: float | None, stderr: float | None) -> list[float] | None:
    if estimate is None or stderr is None:
        return None
    return [estimate - Z95 * stderr, estimate + Z95 * stderr]


def cell_outcomes(
    payload: Mapping[str, Any], task_ids: Sequence[str]
) -> dict[str, dict[str, Any]] | None:
    """Per-task outcome of one evaluation run, or None when it has no health rows (dry run)."""
    health = payload.get("health")
    rows = health.get("rows") if isinstance(health, Mapping) else None
    if not isinstance(rows, list):
        return None
    outcomes: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        scored = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("task_id") == task_id
            and is_scored_row(row)
            and _finite(row.get("reward"))
        ]
        if scored:
            reward = float(scored[-1]["reward"])
            outcomes[task_id] = {
                "reward": reward,
                "passed": reward >= PASS_THRESHOLD,
                "infra_error": False,
            }
        else:
            outcomes[task_id] = {"reward": None, "passed": None, "infra_error": True}
    return outcomes


def _per_task(
    task_ids: Sequence[str], trials: Sequence[Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    per_task = {}
    for task_id in task_ids:
        valid = [trial[task_id] for trial in trials if not trial[task_id]["infra_error"]]
        passes = [1.0 if cell["passed"] else 0.0 for cell in valid]
        per_task[task_id] = {
            "scored_trials": len(valid),
            "pass_mean": _mean(passes),
            "pass_variance": _sample_variance(passes),
            "reward_mean": _mean([float(cell["reward"]) for cell in valid]),
        }
    return per_task


def arm_summary(
    task_ids: Sequence[str], trials: Sequence[Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Pass rate of one arm (baseline or final) on one suite over all its trials."""
    per_task = _per_task(task_ids, trials)
    scored = [task_id for task_id in task_ids if per_task[task_id]["scored_trials"]]
    means = [per_task[task_id]["pass_mean"] for task_id in scored]
    variance = _sample_variance(means)
    pass_rate = _mean(means)
    stderr = math.sqrt(variance / len(means)) if variance is not None else None
    trial_pass_rates = []
    for trial in trials:
        cells = [trial[task_id] for task_id in task_ids if not trial[task_id]["infra_error"]]
        trial_pass_rates.append(
            _mean([1.0 if cell["passed"] else 0.0 for cell in cells])
        )
    return {
        "task_count": len(task_ids),
        "scored_task_count": len(scored),
        "unscored_task_ids": [task_id for task_id in task_ids if task_id not in scored],
        "cells": len(task_ids) * len(trials),
        "infra_error_cells": sum(
            1 for trial in trials for task_id in task_ids if trial[task_id]["infra_error"]
        ),
        "pass_rate": pass_rate,
        "stderr": stderr,
        "ci95": _ci95(pass_rate, stderr),
        "mean_reward": _mean(
            [per_task[task_id]["reward_mean"] for task_id in scored]
        ),
        "trial_pass_rates": trial_pass_rates,
    }


def paired_delta(
    task_ids: Sequence[str],
    baseline: Sequence[Mapping[str, Mapping[str, Any]]],
    final: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Δ = final - baseline pass rate over the tasks scored in both arms, with its SE."""
    before = _per_task(task_ids, baseline)
    after = _per_task(task_ids, final)
    paired = [
        task_id
        for task_id in task_ids
        if before[task_id]["scored_trials"] and after[task_id]["scored_trials"]
    ]
    differences = [
        after[task_id]["pass_mean"] - before[task_id]["pass_mean"] for task_id in paired
    ]
    n = len(paired)
    delta = _mean(differences)
    total_variance = _sample_variance(differences)
    stderr = math.sqrt(total_variance / n) if total_variance is not None else None
    trial_variance = None
    if n and all(
        before[task_id]["pass_variance"] is not None
        and after[task_id]["pass_variance"] is not None
        for task_id in paired
    ):
        trial_variance = (
            sum(
                before[task_id]["pass_variance"] / before[task_id]["scored_trials"]
                + after[task_id]["pass_variance"] / after[task_id]["scored_trials"]
                for task_id in paired
            )
            / n
        )
    return {
        "paired_task_count": n,
        "excluded_task_ids": [task_id for task_id in task_ids if task_id not in paired],
        "baseline_pass_rate": _mean([before[task_id]["pass_mean"] for task_id in paired]),
        "final_pass_rate": _mean([after[task_id]["pass_mean"] for task_id in paired]),
        "delta": delta,
        "stderr": stderr,
        "ci95": _ci95(delta, stderr),
        "variance_components": {
            "total": total_variance,
            "trial": trial_variance,
            "task": (
                max(0.0, total_variance - trial_variance)
                if total_variance is not None and trial_variance is not None
                else None
            ),
        },
        "stderr_fixed_tasks": (
            math.sqrt(trial_variance / n) if trial_variance is not None else None
        ),
    }


def _stratified(
    parts: Sequence[tuple[int, float | None, float | None]],
) -> tuple[float | None, float | None]:
    """Combine (n_s, estimate_s, stderr_s) over suites, weighting every task equally."""
    parts = [part for part in parts if part[0] > 0]
    total = sum(part[0] for part in parts)
    if not total or any(part[1] is None for part in parts):
        return None, None
    estimate = sum(n * value for n, value, _ in parts) / total
    if any(part[2] is None for part in parts):
        return estimate, None
    stderr = math.sqrt(sum((n / total) ** 2 * se**2 for n, _, se in parts))
    return estimate, stderr


def _pooled_arm(arms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pass_rate, stderr = _stratified(
        [(arm["scored_task_count"], arm["pass_rate"], arm["stderr"]) for arm in arms]
    )
    return {
        "task_count": sum(arm["task_count"] for arm in arms),
        "scored_task_count": sum(arm["scored_task_count"] for arm in arms),
        "cells": sum(arm["cells"] for arm in arms),
        "infra_error_cells": sum(arm["infra_error_cells"] for arm in arms),
        "pass_rate": pass_rate,
        "stderr": stderr,
        "ci95": _ci95(pass_rate, stderr),
    }


def _pooled_delta(deltas: Sequence[dict[str, Any]]) -> dict[str, Any]:
    parts = [
        (d["paired_task_count"], d["delta"], d["stderr"]) for d in deltas
    ]
    delta, stderr = _stratified(parts)
    baseline, _ = _stratified(
        [(d["paired_task_count"], d["baseline_pass_rate"], None) for d in deltas]
    )
    final, _ = _stratified(
        [(d["paired_task_count"], d["final_pass_rate"], None) for d in deltas]
    )
    return {
        "paired_task_count": sum(d["paired_task_count"] for d in deltas),
        "baseline_pass_rate": baseline,
        "final_pass_rate": final,
        "delta": delta,
        "stderr": stderr,
        "ci95": _ci95(delta, stderr),
    }


def score_report(
    *,
    suites: Sequence[Mapping[str, Any]],
    trials: int,
    baseline_stage: str,
    final_stage: str | None,
) -> dict[str, Any]:
    """Build the `score_v2` block of score.json.

    `suites` holds, per suite in evaluation order: name, dataset, task_ids, and the per-trial
    outcome maps for `baseline` and `final` (each a list with one entry per trial, or None
    when that arm has no results, as in a dry run or when no post-training evaluation ran).
    """
    rows = []
    for suite in suites:
        task_ids = list(suite["task_ids"])
        baseline = suite.get("baseline")
        final = suite.get("final")
        baseline_arm = arm_summary(task_ids, baseline) if baseline else None
        final_arm = arm_summary(task_ids, final) if final else None
        rows.append(
            {
                "name": suite["name"],
                "dataset": suite["dataset"],
                "task_count": len(task_ids),
                "baseline": baseline_arm,
                "final": final_arm,
                "delta": (
                    paired_delta(task_ids, baseline, final)
                    if baseline and final
                    else None
                ),
            }
        )
    baselines = [row["baseline"] for row in rows]
    finals = [row["final"] for row in rows]
    deltas = [row["delta"] for row in rows]
    return {
        "schema_version": 2,
        "metric": "pass@1",
        "pass_threshold": PASS_THRESHOLD,
        "trials": trials,
        "primary_suite": rows[0]["name"] if rows else None,
        "baseline_stage": baseline_stage,
        "final_stage": final_stage,
        "suites": rows,
        "pooled": {
            "task_count": sum(row["task_count"] for row in rows),
            "baseline": (
                _pooled_arm(baselines) if all(arm is not None for arm in baselines) else None
            ),
            "final": (
                _pooled_arm(finals)
                if finals and all(arm is not None for arm in finals)
                else None
            ),
            "delta": (
                _pooled_delta(deltas)
                if deltas and all(delta is not None for delta in deltas)
                else None
            ),
        },
        "stderr_method": STDERR_METHOD,
    }
