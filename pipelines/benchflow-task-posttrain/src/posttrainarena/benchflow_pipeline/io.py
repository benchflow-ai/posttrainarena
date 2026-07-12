"""Small, dependency-free I/O helpers for pipeline orchestration."""

from __future__ import annotations

import inspect
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_task_ids(path: Path) -> list[str]:
    task_ids = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            task_ids.append(value)
    return list(dict.fromkeys(task_ids))


def supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in values.items() if key in parameters}


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return data


def load_score(path: Path) -> float:
    score = load_json(path).get("score")
    if not isinstance(score, int | float) or isinstance(score, bool):
        raise ValueError(f"Missing numeric score in {path}")
    return float(score)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _resolved_command(command: list[str]) -> tuple[list[str], str | None]:
    if command and command[0] == "bench":
        bundled = Path(sys.executable).with_name("bench")
        if bundled.is_file() and os.access(bundled, os.X_OK):
            return [str(bundled), *command[1:]], str(bundled)
    return command, None


class CommandRunner:
    """Execute and record subprocess stages using argv, never shell strings."""

    def __init__(self, *, cwd: Path, dry_run: bool = False) -> None:
        self.cwd = cwd
        self.dry_run = dry_run
        self.commands: list[dict[str, Any]] = []

    def run(
        self,
        name: str,
        command: list[str],
        *,
        check: bool = True,
        env_overrides: dict[str, str] | None = None,
    ) -> int:
        record = {
            "name": name,
            "cwd": str(self.cwd),
            "command": command,
            "shell": shlex.join(command),
        }
        if env_overrides:
            record["env_keys"] = sorted(env_overrides)
        self.commands.append(record)
        print(
            f"[posttrainarena] {name}: {record['shell']}",
            file=sys.stderr,
            flush=True,
        )
        if self.dry_run:
            record["returncode"] = 0
            return 0
        env = None
        if env_overrides:
            env = {**os.environ, **env_overrides}
        executable_command, resolved_executable = _resolved_command(command)
        if resolved_executable:
            record["resolved_executable"] = resolved_executable
        result = subprocess.run(
            executable_command,
            cwd=self.cwd,
            check=False,
            env=env,
        )
        record["returncode"] = result.returncode
        if check and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        return result.returncode
