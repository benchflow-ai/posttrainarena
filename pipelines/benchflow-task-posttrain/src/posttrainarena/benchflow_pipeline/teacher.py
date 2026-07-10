"""Verified teacher-trajectory collection using BenchFlow's exact tools."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .io import write_json


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final answer for verification.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported provider object: {type(value)!r}")


def _exchange(
    *,
    model: str,
    request_messages: list[dict[str, Any]],
    response: Any,
    duration_ms: int,
) -> dict[str, Any]:
    return {
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": request_messages,
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        },
        "response": {"status_code": 200, "body": _as_dict(response)},
        "duration_ms": duration_ms,
    }


def write_trajectory(rollout_dir: Path, exchanges: list[dict[str, Any]]) -> Path:
    path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in exchanges)
    )
    return path


def collect_verified_teacher_rollouts(
    *,
    config: PipelineConfig,
    tasks_dir: Path,
    task_ids: list[str],
    jobs_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    from benchflow.integrations.trl import (
        BashHarnessConfig,
        BenchFlowSpec,
        benchflow_environment_reward,
    )
    from openai import OpenAI

    api_key = os.environ.get(config.teacher.api_key_env)
    base_url = os.environ.get(config.teacher.base_url_env)
    if not api_key or not base_url:
        raise RuntimeError(
            "Missing teacher credentials: "
            f"{config.teacher.api_key_env} and {config.teacher.base_url_env}"
        )
    client = OpenAI(api_key=api_key, base_url=base_url)
    spec = BenchFlowSpec(
        tasks_dir=tasks_dir,
        include_tasks=task_ids,
        bash_harness=BashHarnessConfig(
            environment=config.runtime.environment,
            sandbox_user=config.runtime.sandbox_user,
            jobs_dir=jobs_dir,
            reset_message=(
                "Solve the task using run_bash to inspect the workspace. "
                "Call submit exactly once with only the final answer."
            ),
            bash_timeout_sec=config.runtime.bash_timeout_sec,
            max_output_chars=config.runtime.max_output_chars,
        ),
    )
    rows = {row["benchflow_task_id"]: row for row in spec.train_dataset_rows}
    attempts: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for task_id in task_ids:
        for attempt in range(1, config.teacher.max_attempts + 1):
            env = spec.environment_factory()
            row = rows[task_id]
            reset_message = env.reset(**row)
            if env.rollout_dir is None:
                raise RuntimeError("BenchFlow did not create a rollout directory")
            messages = [dict(message) for message in row["prompt"]]
            if reset_message:
                messages.append({"role": "user", "content": reset_message})
            exchanges: list[dict[str, Any]] = []
            submitted = False
            error: str | None = None
            try:
                for _ in range(config.runtime.max_tool_calling_iterations):
                    request_messages = [dict(message) for message in messages]
                    started = time.monotonic()
                    response = client.chat.completions.create(
                        model=config.teacher.model,
                        messages=request_messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=config.teacher.temperature,
                        max_tokens=config.teacher.max_tokens,
                    )
                    exchanges.append(
                        _exchange(
                            model=config.teacher.model,
                            request_messages=request_messages,
                            response=response,
                            duration_ms=round((time.monotonic() - started) * 1000),
                        )
                    )
                    assistant = _as_dict(response.choices[0].message)
                    assistant.setdefault("role", "assistant")
                    assistant.setdefault("content", "")
                    messages.append(assistant)
                    tool_calls = assistant.get("tool_calls") or []
                    if not tool_calls:
                        messages.append(
                            {
                                "role": "user",
                                "content": "Call run_bash to continue, or submit the final answer.",
                            }
                        )
                        continue
                    for raw_call in tool_calls:
                        call = _as_dict(raw_call)
                        function = call.get("function", {})
                        name = function.get("name")
                        arguments = function.get("arguments", "{}")
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError(
                                f"Arguments for {name!r} must be an object"
                            )
                        if name == "run_bash":
                            output = env.run_bash(str(arguments.get("command", "")))
                        elif name == "submit":
                            output = env.submit(str(arguments.get("answer", "")))
                            submitted = True
                        else:
                            output = f"unsupported tool: {name}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id"),
                                "content": output,
                            }
                        )
                        if submitted:
                            break
                    if submitted:
                        exchanges.append(
                            _exchange(
                                model=config.teacher.model,
                                request_messages=[
                                    dict(message) for message in messages
                                ],
                                response={
                                    "choices": [
                                        {
                                            "message": {
                                                "role": "assistant",
                                                "content": "Submission verified by the environment.",
                                            }
                                        }
                                    ]
                                },
                                duration_ms=0,
                            )
                        )
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if not submitted:
                    benchflow_environment_reward([None], environments=[env])
                trajectory_path = write_trajectory(env.rollout_dir, exchanges)
            result = {
                "task_id": task_id,
                "attempt": attempt,
                "reward": float(env.reward),
                "submitted": submitted,
                "rollout_dir": str(env.rollout_dir),
                "trajectory": str(trajectory_path),
                "exchange_count": len(exchanges),
                "error": error,
            }
            attempts.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if result["reward"] == 1.0:
                verified.append(result)
                break
    manifest = {
        "teacher_model": config.teacher.model,
        "requested_task_count": len(task_ids),
        "verified_count": len(verified),
        "attempts": attempts,
        "verified": verified,
    }
    write_json(manifest_path, manifest)
    if len(verified) < config.teacher.min_verified:
        raise RuntimeError(
            f"Only {len(verified)} verified trajectories; "
            f"required {config.teacher.min_verified}"
        )
    return manifest
