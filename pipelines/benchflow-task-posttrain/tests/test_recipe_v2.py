from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.pipeline import Pipeline
from posttrainarena.benchflow_pipeline.sampler import prompts_per_step


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qwen3.5-35b-a3b-terminal-grpo-v2.toml"


def _bundle(tmp_path: Path, *, train_tasks: int = 256, text: str | None = None) -> Path:
    """The template inside a bundle-like directory with placeholder task lists."""
    lists = tmp_path / "task-lists"
    lists.mkdir(parents=True)
    (lists / "train.txt").write_text("".join(f"task_{i:06d}\n" for i in range(train_tasks)))
    (lists / "tb2-86.txt").write_text("".join(f"tb2-{i:02d}\n" for i in range(86)))
    (lists / "lhtb-38.txt").write_text("".join(f"lhtb-{i:02d}\n" for i in range(38)))
    path = tmp_path / "config.toml"
    if text is None:
        shutil.copy(TEMPLATE, path)
    else:
        path.write_text(text)
    return path


def test_recipe_v2_template_is_valid_in_a_bundle(tmp_path: Path) -> None:
    config = load_config(_bundle(tmp_path))

    assert config.model == "Qwen/Qwen3.5-35B-A3B"
    assert len(config.model_revision) == 40
    assert [suite.name for suite in config.suites] == ["tb2", "lhtb"]
    assert all(len(suite.dataset.revision) == 40 for suite in config.suites)
    assert config.evaluation.trials == 3
    assert config.grpo.task_sampler == "cover"
    assert config.grpo.rollout_failure_policy == "mask"
    assert config.grpo.gradient_accumulation_steps == config.grpo.generation_batch_size
    assert prompts_per_step(config) == 8
    assert config.grpo.max_steps * prompts_per_step(config) == 256
    assert config.grpo.lora_target_parameters == ()


def test_recipe_v2_dry_run_covers_every_suite_trial_and_task(tmp_path: Path) -> None:
    config = load_config(_bundle(tmp_path))
    pipeline = Pipeline(config, run_name="v2", dry_run=True)

    summary = pipeline.run()

    names = [command["name"] for command in summary["commands"]]
    assert [name for name in names if name.startswith("posttrain_eval")] == [
        "posttrain_eval",
        "posttrain_eval.tb2.t02",
        "posttrain_eval.tb2.t03",
        "posttrain_eval.lhtb.t01",
        "posttrain_eval.lhtb.t02",
        "posttrain_eval.lhtb.t03",
    ]
    provenance = json.loads((pipeline.layout.reports / "provenance.json").read_text())
    assert provenance["sampler"]["task_sampler"] == "cover"
    assert [entry["name"] for entry in provenance["datasets"]] == ["train", "tb2", "lhtb"]


def test_recipe_v2_refuses_collections_its_budget_cannot_cover(tmp_path: Path) -> None:
    config = load_config(_bundle(tmp_path, train_tasks=257))

    with pytest.raises(ValueError, match="cannot cover the 257 training tasks"):
        Pipeline(config, run_name="too-big", dry_run=True)


def test_recipe_v2_expert_lora_targets_parse(tmp_path: Path) -> None:
    text = TEMPLATE.read_text().replace(
        '# lora_target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]',
        'lora_target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]',
    )
    config = load_config(_bundle(tmp_path, text=text))

    assert config.grpo.lora_target_parameters == (
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    )
