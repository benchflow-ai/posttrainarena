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
    assert plan["harness"] == {
        "agent": "opencode",
        "skill_mode": "no-skill",
        "usage_tracking": "required",
        "external_directory_allow": ("/home/user/input/**",),
        "deny_bash_patterns": ("*<<*",),
        "concurrency": 1,
        "sandbox_setup_timeout_sec": 300,
        "agent_idle_timeout_sec": 300,
        "agent_timeout_sec": 900,
        "reasoning_effort": None,
    }
    assert plan["evaluation"] == {
        "base_model_env": "BENCHFLOW_BASE_MODEL",
        "student_model_env": "BENCHFLOW_ADAPTER_MODEL",
        "base_url_env": "BENCHFLOW_PROVIDER_BASE_URL",
        "control_url_env": "BENCHFLOW_MODEL_BRIDGE_CONTROL_URL",
        "api_key_env": "BENCHFLOW_PROVIDER_API_KEY",
    }
    assert plan["harness_migration"] == {
        "applied_stages": ["teacher", "evaluation", "grpo"],
        "pending_stages": [],
    }
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
    assert saved["harness"]["agent"] == "opencode"
    assert saved["teacher"]["require_all_tasks"] is True
    assert saved["sft"]["lora_r"] == config.sft.lora_r
    assert saved["grpo"]["lora_r"] == config.grpo.lora_r
    assert saved["harness_migration"]["applied_stages"] == [
        "teacher",
        "evaluation",
        "grpo",
    ]
    convert = next(
        item
        for item in saved["commands"]
        if item["name"] == "convert_verified_sft_data"
    )
    assert convert["command"][convert["command"].index("--min-reward") + 1] == "1.0"
    assert convert["command"][convert["command"].index("--format") + 1] == "trl-sft"
    assert convert["command"][convert["command"].index("--row-mode") + 1] == (
        "exchange"
    )
    assert convert["command"][convert["command"].index("--tokenizer") + 1] == (
        config.model
    )
    assert convert["command"][convert["command"].index("--max-length") + 1] == (
        str(config.sft.max_length)
    )
    assert len(saved["train_task_ids"]) == 15
    assert {item["name"] for item in saved["commands"]} >= {
        "snapshot_train_tasks",
        "baseline_eval",
        "collect_verified_teacher_rollouts",
        "train_sft",
        "sync_sft_endpoint",
        "sft_eval",
        "grpo_gate_eval",
        "sync_grpo_endpoint",
        "compare_eval_lift",
    }
    evaluation_commands = [
        item
        for item in saved["commands"]
        if item["name"] in {"baseline_eval", "sft_eval", "grpo_gate_eval"}
    ]
    assert all(
        item["command"][item["command"].index("--agent") + 1] == "opencode"
        for item in evaluation_commands
    )


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
    gate_task_ids = [
        gate["command"][index + 1]
        for index, value in enumerate(gate["command"])
        if value == "--include"
    ]

    assert grpo["model"] == config.model
    assert grpo["call"] == "grpo.train_grpo"
    sync = next(
        item for item in result["commands"] if item["name"] == "sync_grpo_endpoint"
    )
    assert sync["checkpoint"].endswith("/checkpoints/grpo-merged")
    assert (
        gate_task_ids
        == Pipeline(config, run_name="task-list", dry_run=True).train_task_ids[
            : config.grpo.gate_task_count
        ]
    )
    assert result["grpo_planned"] is True
    assert result["grpo_ran"] is False
    assert result["checkpoints"]["grpo_adapter"].endswith("/checkpoints/grpo-adapter")
    assert result["checkpoints"]["grpo_merged"].endswith("/checkpoints/grpo-merged")


def test_grpo_run_policy_can_force_zero_reward_training(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    gated = Pipeline(
        replace(config, output_root=tmp_path), run_name="gated", dry_run=False
    )
    forced = Pipeline(
        replace(
            config,
            output_root=tmp_path,
            grpo=replace(config.grpo, run_policy="always"),
        ),
        run_name="forced",
        dry_run=False,
    )

    assert gated._should_run_grpo(0.0) is False
    assert forced._should_run_grpo(0.0) is True


def test_resumed_grpo_restarts_stage_instead_of_reusing_rollouts(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(
        config,
        run_name="resume-grpo",
        dry_run=True,
        resume=True,
    )
    stale = pipeline.layout.jobs / "grpo-train" / "stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    stale_adapter = pipeline.layout.grpo_adapter / "stale.json"
    stale_adapter.parent.mkdir(parents=True)
    stale_adapter.write_text("{}")
    stale_merged = pipeline.layout.grpo_merged / "stale.json"
    stale_merged.parent.mkdir(parents=True)
    stale_merged.write_text("{}")

    pipeline._train_grpo(
        input_model=config.model,
        output_model=str(pipeline.layout.grpo_merged),
    )

    assert not stale.exists()
    assert not stale_adapter.exists()
    assert not stale_merged.exists()
    assert pipeline.runner.commands[-1]["resume_policy"] == "restart-stage"
