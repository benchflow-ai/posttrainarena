"""OpenCode evaluation through BenchFlow's production rollout CLI."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

from .config import PipelineConfig
from .io import CommandRunner, load_json, write_json

ServedModelRole = Literal["auto", "base", "student"]


def _environment_value(name: str, *, label: str, required: bool) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if not required:
        return f"${{{name}}}"
    raise RuntimeError(f"Missing OpenCode {label}: {name}")


def served_model(
    config: PipelineConfig,
    model: str,
    *,
    required: bool = True,
    role: ServedModelRole = "auto",
) -> str:
    if role not in {"auto", "base", "student"}:
        raise ValueError(f"Unknown served model role: {role}")
    resolved_role = "base" if role == "auto" and model == config.model else role
    if resolved_role == "auto":
        resolved_role = "student"
    env_name = (
        config.evaluation.base_model_env
        if resolved_role == "base"
        else config.evaluation.student_model_env
    )
    resolved = _environment_value(
        env_name,
        label="served model",
        required=required,
    )
    if required and "/" not in resolved:
        raise RuntimeError(
            f"OpenCode served model from {env_name} must use provider/model format"
        )
    return resolved


def evaluation_env(
    config: PipelineConfig,
    *,
    required: bool = True,
) -> dict[str, str]:
    return {
        "BENCHFLOW_PROVIDER_BASE_URL": _environment_value(
            config.evaluation.base_url_env,
            label="evaluation endpoint",
            required=required,
        ),
        "BENCHFLOW_PROVIDER_API_KEY": _environment_value(
            config.evaluation.api_key_env,
            label="evaluation API key",
            required=required,
        ),
    }


def opencode_config_env(config: PipelineConfig) -> str:
    content = {
        "permission": {
            "external_directory": {
                path: "allow" for path in config.harness.external_directory_allow
            },
            "bash": {
                **{pattern: "deny" for pattern in config.harness.deny_bash_patterns},
            },
        },
    }
    return "OPENCODE_CONFIG_CONTENT=" + json.dumps(
        content,
        separators=(",", ":"),
    )


def build_evaluation_command(
    *,
    config: PipelineConfig,
    model: str,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    health_path: Path,
    task_manifest_path: Path,
    run_config_path: Path,
    require_environment: bool = True,
    capture_token_logprobs: bool = False,
    model_role: ServedModelRole = "auto",
) -> list[str]:
    if not task_ids:
        raise ValueError("OpenCode evaluation requires at least one task")
    command = [
        "bench",
        "eval",
        "run",
        "--tasks-dir",
        str(tasks_dir),
        "--agent",
        config.harness.agent,
        "--model",
        served_model(
            config,
            model,
            required=require_environment,
            role=model_role,
        ),
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
        "--health-summary-out",
        str(health_path),
        "--task-manifest-out",
        str(task_manifest_path),
        "--run-config-out",
        str(run_config_path),
        "--jobs-dir",
        str(jobs_dir),
        "--agent-env",
        opencode_config_env(config),
    ]
    if config.runtime.sandbox_user is not None:
        command.extend(["--sandbox-user", config.runtime.sandbox_user])
    if config.harness.reasoning_effort:
        command.extend(["--reasoning-effort", config.harness.reasoning_effort])
    if capture_token_logprobs:
        command.extend(
            [
                "--agent-env",
                "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1",
            ]
        )
    for task_id in task_ids:
        command.extend(["--include", task_id])
    return command


def _ratio(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RuntimeError(f"OpenCode evaluation summary has no numeric {key}")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise RuntimeError(f"OpenCode evaluation {key} is invalid: {value!r}")
    return result


def _count(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"OpenCode evaluation summary has no valid {key}")
    return value


TIMEOUT_MARKERS = ("wall-clock budget", "idle timeout", "timed out")


def is_timeout_error(row: Mapping[str, Any]) -> bool:
    """An agent that ran out of its wall-clock or idle budget produced a scored failure (reward 0), not an infrastructure error.

    Terminal-Bench and Tmax count such attempts as failures; treating them as unhealthy made every
    evaluation with one slow task abort (phase-2 run r23 lost its baseline to nine 900 s timeouts).
    """
    error = row.get("error")
    if error is None:
        return False
    category = row.get("error_category")
    if category is not None:
        return str(category) == "timeout"
    return any(marker in str(error).lower() for marker in TIMEOUT_MARKERS)


def is_scored_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("scored") is True
        and (row.get("error") is None or is_timeout_error(row))
        and row.get("verifier_error") is None
        and row.get("valid_llm_trajectory") is True
    )


def max_infra_errors_for(config: PipelineConfig, task_count: int) -> int:
    """Number of infrastructure-errored tasks an evaluation may carry (counted as failures)."""
    return int(math.ceil(config.harness.max_infra_error_fraction * task_count))


def load_summary(
    *,
    jobs_dir: Path,
    health_path: Path,
    expected_tasks: int,
    expected_task_ids: list[str] | None = None,
    max_infra_errors: int = 0,
) -> dict[str, Any]:
    summary_path = jobs_dir / "summary.json"
    summary = load_json(summary_path)
    if _count(summary, "total") != expected_tasks:
        raise RuntimeError(
            f"OpenCode evaluation produced {summary.get('total')!r} tasks; "
            f"expected {expected_tasks}"
        )
    errored = _count(summary, "errored") + _count(summary, "verifier_errored")
    if errored > max_infra_errors:
        raise RuntimeError(
            "OpenCode evaluation contains agent or verifier errors"
            + (f" ({errored} > {max_infra_errors} tolerated)" if max_infra_errors else "")
        )
    # Errored tasks carry no usage telemetry; coverage must be complete for every other task.
    if _ratio(summary, "telemetry_coverage") < (expected_tasks - errored) / expected_tasks - 1e-9:
        raise RuntimeError("OpenCode evaluation telemetry coverage is incomplete")
    health = load_json(health_path)
    infra_error_tasks: list[str] = []
    if expected_task_ids is not None:
        rows = health.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError("OpenCode evaluation health summary has no valid rows")
        missing = []
        for task_id in expected_task_ids:
            valid = any(row.get("task_id") == task_id and is_scored_row(row) for row in rows)
            if valid:
                continue
            attempted = any(
                row.get("task_id") == task_id
                and (row.get("error") is not None or row.get("verifier_error") is not None)
                for row in rows
            )
            if attempted and len(infra_error_tasks) < max_infra_errors:
                infra_error_tasks.append(task_id)
            else:
                missing.append(task_id)
        if missing:
            raise RuntimeError(
                "OpenCode evaluation has no healthy scored rollout for: "
                + ", ".join(missing)
            )
    else:
        if _count(health, "total_rows") != expected_tasks:
            raise RuntimeError(
                f"OpenCode evaluation health summary has "
                f"{health.get('total_rows')!r} rows; expected {expected_tasks}"
            )
        if _count(health, "scored_rows") != expected_tasks:
            raise RuntimeError(
                f"OpenCode evaluation health summary has "
                f"scored_rows={health.get('scored_rows')!r}; expected {expected_tasks}"
            )
        for key in (
            "unscored_rows",
            "missing_llm_trajectory",
            "malformed_llm_trajectory",
        ):
            value = _count(health, key)
            if value != 0:
                raise RuntimeError(
                    f"OpenCode evaluation health summary has {key}={value!r}"
                )
    if infra_error_tasks:
        # Errored tasks count as failures: passes over every expected task.
        passed = summary.get("passed")
        if not isinstance(passed, int) or isinstance(passed, bool) or passed < 0:
            raise RuntimeError("OpenCode evaluation summary has no integer passed count")
        score = passed / expected_tasks
    else:
        score_key = (
            "score_excl_errors_ratio"
            if "score_excl_errors_ratio" in summary
            else "score_ratio"
        )
        score = _ratio(summary, score_key)
    return {
        "score": score,
        "summary": summary,
        "summary_path": str(summary_path),
        "health": health,
        "health_path": str(health_path),
        "infra_error_tasks": infra_error_tasks,
    }


def evaluate(
    *,
    config: PipelineConfig,
    runner: CommandRunner,
    stage: str,
    model: str,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    metrics_path: Path,
    capture_token_logprobs: bool = False,
    model_role: ServedModelRole = "auto",
    policy_sha256: str | None = None,
) -> dict[str, Any]:
    require_environment = not runner.dry_run
    health_path = metrics_path.with_name(f"{metrics_path.stem}_health.json")
    task_manifest_path = metrics_path.with_name(
        f"{metrics_path.stem}_task_manifest.json"
    )
    run_config_path = metrics_path.with_name(f"{metrics_path.stem}_run_config.json")
    # `bench eval run` exits non-zero when any task errored; the summary decides whether
    # that is tolerable (see load_summary / harness.max_infra_error_fraction).
    returncode = runner.run(
        stage,
        build_evaluation_command(
            config=config,
            model=model,
            tasks_dir=tasks_dir,
            task_ids=task_ids,
            jobs_dir=jobs_dir,
            health_path=health_path,
            task_manifest_path=task_manifest_path,
            run_config_path=run_config_path,
            require_environment=require_environment,
            capture_token_logprobs=capture_token_logprobs,
            model_role=model_role,
        ),
        env_overrides=evaluation_env(config, required=require_environment),
        check=False,
    )
    if not runner.dry_run and returncode and not (jobs_dir / "summary.json").is_file():
        raise RuntimeError(
            f"{stage}: bench eval run exited with {returncode} and produced no summary"
        )
    if runner.dry_run:
        return {
            "mode": "eval",
            "harness": config.harness.agent,
            "model": model,
            "served_model": served_model(
                config,
                model,
                required=False,
                role=model_role,
            ),
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "score": None,
            "jobs_dir": str(jobs_dir),
            "capture_token_logprobs": capture_token_logprobs,
            "policy_sha256": policy_sha256,
        }
    loaded = load_summary(
        jobs_dir=jobs_dir,
        health_path=health_path,
        expected_tasks=len(task_ids),
        expected_task_ids=task_ids,
        max_infra_errors=max_infra_errors_for(config, len(task_ids)),
    )
    payload = {
        "mode": "eval",
        "harness": config.harness.agent,
        "model": model,
        "served_model": served_model(config, model, role=model_role),
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "score": loaded["score"],
        "summary": loaded["summary"],
        "health": loaded["health"],
        "jobs_dir": str(jobs_dir),
        "health_path": str(health_path),
        "task_manifest_path": str(task_manifest_path),
        "run_config_path": str(run_config_path),
        "capture_token_logprobs": capture_token_logprobs,
        "policy_sha256": policy_sha256,
    }
    write_json(metrics_path, payload)
    return payload
