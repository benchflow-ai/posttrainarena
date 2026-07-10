from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_is_valid_and_pinned() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")

    assert config.model == "Qwen/Qwen3-4B"
    assert len(config.train_dataset.revision) == 40
    assert len(config.eval_dataset.revision) == 40
    assert config.teacher.min_verified == 15
    assert config.runtime.num_generations == 2
    assert config.grpo.run_policy == "on_reward"


def test_config_accepts_always_grpo_run_policy(tmp_path: Path) -> None:
    source = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "always.toml"
    config_path.write_text(
        source.read_text().replace(
            'run_policy = "on_reward"', 'run_policy = "always"'
        )
    )

    config = load_config(config_path)

    assert config.grpo.run_policy == "always"


def test_config_rejects_unknown_grpo_run_policy() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, grpo=replace(config.grpo, run_policy="unconditional"))  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="grpo.run_policy must be on_reward or always",
    ):
        config.validate()


def test_config_rejects_conflicting_sandbox_aliases() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        runtime=replace(config.runtime, environment="docker", sandbox="daytona"),
    )

    with pytest.raises(ValueError, match="runtime.sandbox conflicts"):
        config.validate()


def test_openenv_url_requires_openenv_integration() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        runtime=replace(config.runtime, openenv_url="http://localhost:8000"),
    )

    with pytest.raises(ValueError, match="openenv_url requires"):
        config.validate()


def test_config_rejects_missing_task_list(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        """
[model]
id = "model"
[train_dataset]
repo_id = "train"
revision = "abc"
task_list = "missing-train.txt"
[eval_dataset]
repo_id = "eval"
revision = "def"
task_list = "missing-eval.txt"
"""
    )

    with pytest.raises(ValueError, match="does not exist"):
        load_config(config)
