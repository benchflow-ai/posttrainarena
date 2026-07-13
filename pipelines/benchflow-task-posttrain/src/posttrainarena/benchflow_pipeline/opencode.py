"""OpenCode evaluation through BenchFlow's production rollout CLI."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
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
        }
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


def load_summary(
    *,
    jobs_dir: Path,
    health_path: Path,
    expected_tasks: int,
) -> dict[str, Any]:
    summary_path = jobs_dir / "summary.json"
    summary = load_json(summary_path)
    if _count(summary, "total") != expected_tasks:
        raise RuntimeError(
            f"OpenCode evaluation produced {summary.get('total')!r} tasks; "
            f"expected {expected_tasks}"
        )
    if _count(summary, "errored") or _count(summary, "verifier_errored"):
        raise RuntimeError("OpenCode evaluation contains agent or verifier errors")
    if _ratio(summary, "telemetry_coverage") < 1.0:
        raise RuntimeError("OpenCode evaluation telemetry coverage is incomplete")
    health = load_json(health_path)
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
        "zero_tool_rows",
        "missing_llm_trajectory",
        "malformed_llm_trajectory",
    ):
        value = _count(health, key)
        if value != 0:
            raise RuntimeError(
                f"OpenCode evaluation health summary has {key}={value!r}"
            )
    score_key = (
        "score_excl_errors_ratio"
        if "score_excl_errors_ratio" in summary
        else "score_ratio"
    )
    return {
        "score": _ratio(summary, score_key),
        "summary": summary,
        "summary_path": str(summary_path),
        "health": health,
        "health_path": str(health_path),
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
) -> dict[str, Any]:
    require_environment = not runner.dry_run
    health_path = metrics_path.with_name(f"{metrics_path.stem}_health.json")
    task_manifest_path = metrics_path.with_name(
        f"{metrics_path.stem}_task_manifest.json"
    )
    run_config_path = metrics_path.with_name(f"{metrics_path.stem}_run_config.json")
    runner.run(
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
        }
    loaded = load_summary(
        jobs_dir=jobs_dir,
        health_path=health_path,
        expected_tasks=len(task_ids),
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
    }
    write_json(metrics_path, payload)
    return payload
