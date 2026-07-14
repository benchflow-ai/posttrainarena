"""Verified teacher trajectories collected through BenchFlow and OpenCode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io import CommandRunner, write_json
from .opencode import opencode_config_env


def build_teacher_command(
    *,
    config: PipelineConfig,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    task_manifest_path: Path,
    run_config_path: Path,
    health_path: Path,
) -> list[str]:
    if not task_ids:
        raise ValueError("OpenCode teacher collection requires at least one task")
    command = [
        "bench",
        "eval",
        "run",
        "--tasks-dir",
        str(tasks_dir),
        "--agent",
        config.harness.agent,
        "--model",
        config.teacher.model,
        "--sandbox",
        config.sandbox,
        "--skill-mode",
        config.harness.skill_mode,
        "--concurrency",
        str(config.harness.concurrency),
        "--usage-tracking",
        config.harness.usage_tracking,
        "--sandbox-setup-timeout",
        str(config.harness.sandbox_setup_timeout_sec),
        "--agent-idle-timeout",
        str(config.harness.agent_idle_timeout_sec),
        "--config-override",
        json.dumps(
            {"agent": {"timeout_sec": config.harness.agent_timeout_sec}},
            separators=(",", ":"),
        ),
        "--expected-tasks",
        str(len(task_ids)),
        "--task-manifest-out",
        str(task_manifest_path),
        "--run-config-out",
        str(run_config_path),
        "--health-summary-out",
        str(health_path),
        "--jobs-dir",
        str(jobs_dir),
        "--agent-env",
        opencode_config_env(config),
    ]
    if config.runtime.sandbox_user is not None:
        command.extend(["--sandbox-user", config.runtime.sandbox_user])
    if config.harness.reasoning_effort:
        command.extend(["--reasoning-effort", config.harness.reasoning_effort])
    for task_id in task_ids:
        command.extend(["--include", task_id])
    return command


def _raw_results_row(rollout_dir: Path) -> dict[str, Any] | None:
    path = rollout_dir / "results.jsonl"
    if not path.is_file():
        return None
    rows = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    return rows[0]


def _results_row(rollout_dir: Path) -> dict[str, Any] | None:
    row = _raw_results_row(rollout_dir)
    if (
        row is None
        or not isinstance(row.get("info"), dict)
        or row["info"].get("training_ready") is not True
        or row.get("is_completed") is not True
        or row.get("is_truncated") is not False
        or row.get("stop_condition") != "agent_completed"
        or row.get("error") is not None
    ):
        return None
    return row


def _token_total(row: dict[str, Any]) -> float | None:
    usage = row.get("token_usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int | float) and not isinstance(total, bool):
        return float(total)
    split_total = 0.0
    found_split = False
    for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            found_split = True
            split_total += float(value)
    return split_total if found_split else None


def _training_ready(rollout_dir: Path) -> bool:
    row = _results_row(rollout_dir)
    if row is None:
        return False
    total = _token_total(row)
    return total is not None and total > 0


def _valid_jsonl(path: Path) -> bool:
    if not path.is_file():
        return False
    rows: list[dict[str, Any]] = []
    allowed_types = {"user_message", "agent_message", "agent_thought", "tool_call"}
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("type") not in allowed_types:
                    return False
                event_type = row["type"]
                if event_type in {"user_message", "agent_message", "agent_thought"}:
                    if not isinstance(row.get("text"), str) or not row["text"].strip():
                        return False
                elif event_type == "tool_call":
                    content = row.get("content")
                    if (
                        not isinstance(row.get("tool_call_id"), str)
                        or not row["tool_call_id"]
                        or not isinstance(row.get("title"), str)
                        or not row["title"]
                        or not isinstance(row.get("kind"), str)
                        or not row["kind"]
                        or row.get("status")
                        not in {"pending", "in_progress", "completed", "failed"}
                        or not isinstance(content, list)
                        or not content
                    ):
                        return False
                    for item in content:
                        if (
                            not isinstance(item, dict)
                            or not isinstance(item.get("type"), str)
                            or not item["type"]
                        ):
                            return False
                        if item["type"] == "content":
                            nested = item.get("content")
                            if (
                                not isinstance(nested, dict)
                                or not isinstance(nested.get("type"), str)
                                or not nested["type"]
                            ):
                                return False
                            if nested["type"] == "text" and not isinstance(
                                nested.get("text"), str
                            ):
                                return False
                rows.append(row)
    except (OSError, json.JSONDecodeError):
        return False
    event_types = {str(row["type"]) for row in rows}
    return {
        "user_message",
        "tool_call",
    }.issubset(event_types) and bool(event_types & {"agent_message", "agent_thought"})


def _select_verified(
    *,
    config: PipelineConfig,
    jobs_dir: Path,
    task_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from benchflow.eval_artifacts import build_health_summary

    health = build_health_summary(jobs_dir)
    attempts: list[dict[str, Any]] = []
    for row in health["rows"]:
        rollout_dir = Path(str(row.get("rollout_dir") or ""))
        reward = row.get("reward")
        training_ready = _training_ready(rollout_dir)
        raw_results_row = _raw_results_row(rollout_dir)
        total_tokens = (
            _token_total(raw_results_row) if raw_results_row is not None else None
        )
        acp_trajectory_ready = _valid_jsonl(
            rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        )
        eligible = (
            row.get("task_id") in task_ids
            and row.get("scored") is True
            and isinstance(reward, int | float)
            and not isinstance(reward, bool)
            and float(reward) >= config.teacher.min_reward
            and row.get("valid_llm_trajectory") is True
            and int(row.get("tool_calls") or 0) > 0
            and int(row.get("tool_calls") or 0)
            <= config.teacher.max_accepted_tool_calls
            and row.get("error") is None
            and row.get("verifier_error") is None
            and training_ready
            and acp_trajectory_ready
            and total_tokens is not None
            and total_tokens <= config.teacher.max_accepted_total_tokens
        )
        enriched = {
            **row,
            "training_ready": training_ready,
            "acp_trajectory_ready": acp_trajectory_ready,
            "total_tokens": total_tokens,
            "eligible": eligible,
        }
        attempts.append(enriched)
    return attempts, _select_from_attempts(attempts, task_ids)


def _select_from_attempts(
    attempts: list[dict[str, Any]],
    task_ids: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        grouped.setdefault(str(row.get("task_id")), []).append(row)
    selected = []
    for task_id in task_ids:
        candidates = [row for row in grouped.get(task_id, []) if row["eligible"]]
        if not candidates:
            continue
        winner = max(
            candidates,
            key=lambda row: (
                float(row["reward"]),
                int(row.get("tool_calls") or 0),
                int(row.get("llm_trajectory_rows") or 0),
            ),
        )
        selected.append(
            {
                **winner,
                "selection_reason": "reward-then-tool-calls-then-trajectory-length",
            }
        )
    return selected


def collect_verified_teacher_rollouts(
    *,
    config: PipelineConfig,
    runner: CommandRunner,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    manifest_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    remaining = list(task_ids)
    attempts_so_far: list[dict[str, Any]] = []
    selected_so_far: list[dict[str, Any]] = []
    for attempt in range(1, config.teacher.max_attempts + 1):
        attempt_name = f"attempt-{attempt:02d}"
        attempt_jobs = jobs_dir / attempt_name
        returncode = runner.run(
            (
                "collect_verified_teacher_rollouts"
                if attempt == 1
                else f"collect_verified_teacher_rollouts_{attempt_name}"
            ),
            build_teacher_command(
                config=config,
                tasks_dir=tasks_dir,
                task_ids=remaining,
                jobs_dir=attempt_jobs,
                task_manifest_path=manifest_path.with_name(
                    f"teacher_task_manifest_{attempt_name}.json"
                ),
                run_config_path=manifest_path.with_name(
                    f"teacher_run_config_{attempt_name}.json"
                ),
                health_path=manifest_path.with_name(
                    f"teacher_health_{attempt_name}.json"
                ),
            ),
            check=False,
        )
        if returncode not in {0, 1}:
            raise RuntimeError(
                f"OpenCode teacher attempt failed before rollout completion "
                f"(exit {returncode})"
            )
        if runner.dry_run:
            continue
        new_attempts, _ = _select_verified(
            config=config,
            jobs_dir=attempt_jobs,
            task_ids=remaining,
        )
        attempts_so_far.extend(new_attempts)
        selected_so_far = _select_from_attempts(attempts_so_far, task_ids)
        selected_task_ids = {str(row["task_id"]) for row in selected_so_far}
        over_budget_task_ids = {
            str(row["task_id"])
            for row in attempts_so_far
            if (
                isinstance(row.get("total_tokens"), int | float)
                and float(row["total_tokens"])
                > config.teacher.max_accepted_total_tokens
            )
            or int(row.get("tool_calls") or 0) > config.teacher.max_accepted_tool_calls
        }
        remaining = [
            task_id
            for task_id in task_ids
            if task_id not in selected_task_ids and task_id not in over_budget_task_ids
        ]
        if not remaining:
            break
    if runner.dry_run:
        return {
            "teacher_model": config.teacher.model,
            "teacher_source_model": config.teacher.source_model,
            "teacher_source_revision": config.teacher.source_revision,
            "teacher_source_identity_enforced": False,
            "requested_task_count": len(task_ids),
            "requested_task_ids": task_ids,
            "required_verified_count": (
                len(task_ids)
                if config.teacher.require_all_tasks
                else config.teacher.min_verified
            ),
            "verified_count": None,
            "attempts": [],
            "verified": [],
        }
    attempts = attempts_so_far
    verified = _select_from_attempts(attempts, task_ids)
    selected_dirs = {str(row.get("rollout_dir")) for row in verified}
    requested_task_ids = set(task_ids)
    selection = {
        "schema_version": 1,
        "job_dir": str(jobs_dir),
        "policy": "one-training-ready-per-task",
        "selected_count": len(verified),
        "selected": verified,
        "rejected": [
            row
            for row in attempts
            if row.get("task_id") in requested_task_ids
            and str(row.get("rollout_dir")) not in selected_dirs
        ],
    }
    write_json(selection_path, selection)
    manifest = {
        "teacher_model": config.teacher.model,
        "teacher_source_model": config.teacher.source_model,
        "teacher_source_revision": config.teacher.source_revision,
        "teacher_source_identity_enforced": False,
        "harness": config.harness.agent,
        "requested_task_count": len(task_ids),
        "requested_task_ids": task_ids,
        "required_verified_count": (
            len(task_ids)
            if config.teacher.require_all_tasks
            else config.teacher.min_verified
        ),
        "require_all_tasks": config.teacher.require_all_tasks,
        "verified_count": len(verified),
        "min_reward": config.teacher.min_reward,
        "attempts": attempts,
        "verified": verified,
        "selection_path": str(selection_path),
    }
    write_json(manifest_path, manifest)
    required_verified = (
        len(task_ids)
        if config.teacher.require_all_tasks
        else config.teacher.min_verified
    )
    if len(verified) < required_verified:
        raise RuntimeError(
            f"Only {len(verified)} verified trajectories; required {required_verified}"
        )
    return manifest
