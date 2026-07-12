from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrainarena.benchflow_pipeline.io import CommandRunner


def test_command_runner_can_return_nonzero_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    runner = CommandRunner(cwd=tmp_path)

    returncode = runner.run("teacher", ["bench", "eval", "run"], check=False)

    assert returncode == 1
    assert runner.commands[0]["returncode"] == 1


def test_command_runner_still_raises_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2),
    )
    runner = CommandRunner(cwd=tmp_path)

    with pytest.raises(subprocess.CalledProcessError):
        runner.run("invalid", ["bench", "bad"])


def test_command_runner_dry_run_returns_success_without_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args, **_kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(subprocess, "run", unexpected)
    runner = CommandRunner(cwd=tmp_path, dry_run=True)

    assert runner.run("dry", ["bench", "eval", "run"]) == 0
    assert runner.commands[0]["returncode"] == 0
