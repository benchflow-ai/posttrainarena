from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_plan_exposes_public_stage_contract(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    plan = Pipeline(config, run_name="review", dry_run=True).plan()

    assert plan["train_task_count"] == 15
    assert plan["eval_task_count"] == 2
    assert plan["stages"][0] == "snapshot_train_tasks"
    assert plan["stages"][-1] == "write_score_report"


def test_dry_run_writes_score_schema_without_heavy_dependencies(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(config, run_name="dry-run", dry_run=True)

    result = pipeline.run()
    saved = json.loads((tmp_path / "dry-run/reports/score.json").read_text())

    assert result["score_after_posttrain"] is None
    assert saved["schema_version"] == 1
    assert saved["grpo_planned"] is True
    assert saved["grpo_ran"] is False
    assert len(saved["train_task_ids"]) == 15
    assert {item["name"] for item in saved["commands"]} >= {
        "snapshot_train_tasks",
        "baseline_eval",
        "collect_verified_teacher_rollouts",
        "train_sft",
        "sft_eval",
        "grpo_gate_eval",
        "compare_eval_lift",
    }


def test_pipeline_rejects_train_eval_overlap(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    overlap = tmp_path / "eval.txt"
    overlap.write_text("0060_573_60573328_qa_1\n")
    config = replace(
        config,
        output_root=tmp_path / "runs",
        eval_dataset=replace(config.eval_dataset, task_list=overlap),
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        Pipeline(config, run_name="overlap", dry_run=True)


def test_rl_only_dry_run_uses_base_model(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        output_root=tmp_path,
        sft=replace(config.sft, enabled=False),
        grpo=replace(config.grpo, threshold=0.0),
    )
    result = Pipeline(config, run_name="rl-only", dry_run=True).run()
    grpo = next(item for item in result["commands"] if item["name"] == "train_grpo")
    gate = next(item for item in result["commands"] if item["name"] == "grpo_gate_eval")

    assert grpo["model"] == config.model
    assert (
        gate["task_ids"]
        == Pipeline(config, run_name="task-list", dry_run=True).train_task_ids[
            : config.grpo.gate_task_count
        ]
    )
    assert result["grpo_planned"] is True
    assert result["grpo_ran"] is False
