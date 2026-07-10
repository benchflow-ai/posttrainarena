from __future__ import annotations

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
