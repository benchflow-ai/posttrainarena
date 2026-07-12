from __future__ import annotations

import subprocess
import sys
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


def test_command_runner_passes_env_overrides_without_recording_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs.get("env") or {})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = CommandRunner(cwd=tmp_path)

    runner.run(
        "eval",
        ["bench", "eval", "run"],
        env_overrides={"SECRET_TOKEN": "secret-value"},
    )

    assert captured["SECRET_TOKEN"] == "secret-value"
    assert runner.commands[0]["env_keys"] == ["SECRET_TOKEN"]
    assert "secret-value" not in str(runner.commands[0])


def test_command_runner_prefers_bench_next_to_active_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    bench = bin_dir / "bench"
    python.touch()
    bench.touch()
    bench.chmod(0o755)
    captured: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = CommandRunner(cwd=tmp_path)

    runner.run("eval", ["bench", "eval", "run"])

    assert captured["command"] == [str(bench), "eval", "run"]
    assert runner.commands[0]["command"] == ["bench", "eval", "run"]
    assert runner.commands[0]["resolved_executable"] == str(bench)
