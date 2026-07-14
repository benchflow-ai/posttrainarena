from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.io import directory_sha256
from posttrainarena.benchflow_pipeline.pipeline import (
    Pipeline,
    _sha256,
    _teacher_sources_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def test_plan_exposes_public_stage_contract(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    plan = Pipeline(config, run_name="review", dry_run=True).plan()

    assert plan["train_task_count"] == 15
    assert plan["eval_task_count"] == 2
    assert (
        plan["train_task_ids"]
        == Pipeline(config, run_name="task-ids", dry_run=True).train_task_ids
    )
    assert (
        plan["eval_task_ids"]
        == Pipeline(config, run_name="task-ids", dry_run=True).eval_task_ids
    )
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
        "sync_base_to_vllm": False,
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


def test_pipeline_rejects_same_task_content_under_different_ids(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    train_id = config.train_dataset.task_list.read_text().splitlines()[0]
    eval_id = config.eval_dataset.task_list.read_text().splitlines()[0]
    train_list = tmp_path / "train.txt"
    eval_list = tmp_path / "eval.txt"
    train_list.write_text(train_id + "\n")
    eval_list.write_text(eval_id + "\n")
    config = replace(
        config,
        output_root=tmp_path,
        train_dataset=replace(config.train_dataset, task_list=train_list),
        eval_dataset=replace(config.eval_dataset, task_list=eval_list),
    )
    pipeline = Pipeline(config, run_name="content-overlap", dry_run=False)
    for task_id, root in (
        (train_id, pipeline.layout.train_tasks),
        (eval_id, pipeline.layout.eval_tasks),
    ):
        task_dir = root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.md").write_text(
            "---\n"
            "task:\n"
            f"  name: {task_id}\n"
            "  description: identical\n"
            "---\n"
            "same prompt\n"
        )
        (task_dir / "verifier.py").write_text("print('same')\n")

    with pytest.raises(RuntimeError, match="overlap by canonical content digest"):
        pipeline._validate_task_content_isolation()


def test_resume_validates_snapshot_marker_and_task_bytes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    train_id = config.train_dataset.task_list.read_text().splitlines()[0]
    train_list = tmp_path / "train.txt"
    train_list.write_text(train_id + "\n")
    config = replace(
        config,
        output_root=tmp_path / "runs",
        train_dataset=replace(config.train_dataset, task_list=train_list),
    )
    pipeline = Pipeline(
        config,
        run_name="snapshot-integrity",
        dry_run=True,
        resume=True,
    )
    destination = pipeline.layout.train_tasks
    task_dir = destination / train_id
    task_dir.mkdir(parents=True)
    (task_dir / "task.md").write_text("---\ntask:\n  name: task\n---\nprompt\n")
    marker = destination / ".benchflow-source.json"
    marker.write_text(
        json.dumps(
            {
                "repo": config.train_dataset.repo_id,
                "repo_type": "dataset",
                "path": config.train_dataset.path,
                "requested_revision": config.train_dataset.revision,
                "resolved_revision": config.train_dataset.revision,
                "include_tasks": [train_id],
                "dirty": False,
                "local_path": str(destination),
            }
        )
    )
    pipeline._write_snapshot_integrity(
        label="train",
        dataset=config.train_dataset,
        task_ids=[train_id],
        destination=destination,
        marker=marker,
    )

    pipeline._snapshot(
        "train",
        config.train_dataset,
        [train_id],
        destination,
    )
    assert pipeline.runner.commands == []

    (task_dir / "task.md").write_text("---\ntask:\n  name: renamed-task\n---\nprompt\n")
    with pytest.raises(RuntimeError, match="incompatible task snapshot"):
        pipeline._snapshot(
            "train",
            config.train_dataset,
            [train_id],
            destination,
        )


def test_qwen35_dry_run_syncs_pinned_base_before_baseline(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-soccer-canary.toml")
    config = replace(config, output_root=tmp_path)

    result = Pipeline(config, run_name="base-sync", dry_run=True).run()
    names = [item["name"] for item in result["commands"]]
    sync = next(
        item for item in result["commands"] if item["name"] == "sync_base_endpoint"
    )

    assert names.index("sync_base_endpoint") < names.index("baseline_eval")
    assert sync["reference"] == config.model
    assert sync["revision"] == config.model_revision


def test_resume_revalidates_evaluation_identity_and_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(
        config,
        run_name="resume-eval",
        dry_run=False,
        resume=True,
    )
    task_ids = ["task-a"]
    jobs_dir = pipeline.layout.jobs / "baseline"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "errored": 0,
                "verifier_errored": 0,
                "telemetry_coverage": 1.0,
                "score_excl_errors_ratio": 1.0,
            }
        )
    )
    metrics_path = pipeline.layout.results / "baseline_eval.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.with_name("baseline_eval_health.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task_id": "task-a",
                        "scored": True,
                        "error": None,
                        "verifier_error": None,
                        "valid_llm_trajectory": True,
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("BENCHFLOW_BASE_MODEL", "vllm/base")
    metrics = {
        "mode": "eval",
        "harness": config.harness.agent,
        "model": config.model,
        "served_model": "vllm/base",
        "task_ids": task_ids,
        "task_count": 1,
        "score": 1.0,
        "jobs_dir": str(jobs_dir),
        "capture_token_logprobs": False,
        "policy_sha256": "policy-hash",
    }
    metrics_path.write_text(json.dumps(metrics))

    assert (
        pipeline._load_resumed_evaluation(
            model=config.model,
            task_ids=task_ids,
            jobs_dir=jobs_dir,
            metrics_path=metrics_path,
            policy_sha256="policy-hash",
        )
        == 1.0
    )
    metrics["task_ids"] = ["other-task"]
    metrics_path.write_text(json.dumps(metrics))

    with pytest.raises(RuntimeError, match="incompatible evaluation artifacts"):
        pipeline._load_resumed_evaluation(
            model=config.model,
            task_ids=task_ids,
            jobs_dir=jobs_dir,
            metrics_path=metrics_path,
            policy_sha256="policy-hash",
        )


def test_resume_restarts_incomplete_evaluation_stage(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(
        config,
        run_name="restart-eval",
        dry_run=True,
        resume=True,
    )
    jobs_dir = pipeline.layout.jobs / "baseline"
    jobs_dir.mkdir(parents=True)
    (jobs_dir / "stale.json").write_text("{}")
    metrics_path = pipeline.layout.results / "baseline_eval.json"
    metrics_path.parent.mkdir(parents=True)
    for suffix in ("health", "task_manifest", "run_config"):
        metrics_path.with_name(f"baseline_eval_{suffix}.json").write_text("{}")

    pipeline._evaluate(
        stage="baseline_eval",
        model=config.model,
        tasks_dir=pipeline.layout.eval_tasks,
        task_ids=pipeline.eval_task_ids,
        jobs_dir=jobs_dir,
        metrics_path=metrics_path,
        policy_sha256="policy-hash",
    )

    assert not (jobs_dir / "stale.json").exists()
    assert all(
        not metrics_path.with_name(f"baseline_eval_{suffix}.json").exists()
        for suffix in ("health", "task_manifest", "run_config")
    )
    assert pipeline.runner.commands[-1]["name"] == "baseline_eval"


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


def test_resume_rejects_insufficient_teacher_manifest(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(
        config,
        run_name="resume-teacher",
        dry_run=True,
        resume=True,
    )
    pipeline.layout.reports.mkdir(parents=True)
    (pipeline.layout.reports / "teacher_manifest.json").write_text(
        json.dumps(
            {
                "teacher_model": config.teacher.model,
                "teacher_source_model": config.teacher.source_model,
                "teacher_source_revision": config.teacher.source_revision,
                "requested_task_count": len(pipeline.train_task_ids),
                "requested_task_ids": pipeline.train_task_ids,
                "required_verified_count": len(pipeline.train_task_ids),
                "require_all_tasks": True,
                "verified_count": len(pipeline.train_task_ids) - 1,
            }
        )
    )
    pipeline.layout.teacher_selection.parent.mkdir(parents=True, exist_ok=True)
    pipeline.layout.teacher_selection.write_text(
        json.dumps({"selected_count": len(pipeline.train_task_ids) - 1})
    )

    with pytest.raises(RuntimeError, match="verified_count"):
        pipeline._collect_and_convert_teacher_data()


def test_resume_rejects_fabricated_teacher_selection_count(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(
        config,
        run_name="resume-forged-selection",
        dry_run=True,
        resume=True,
    )
    required = len(pipeline.train_task_ids)
    pipeline.layout.reports.mkdir(parents=True)
    (pipeline.layout.reports / "teacher_manifest.json").write_text(
        json.dumps(
            {
                "teacher_model": config.teacher.model,
                "teacher_source_model": config.teacher.source_model,
                "teacher_source_revision": config.teacher.source_revision,
                "requested_task_count": required,
                "requested_task_ids": pipeline.train_task_ids,
                "required_verified_count": required,
                "require_all_tasks": True,
                "verified_count": required,
                "verified": [],
            }
        )
    )
    pipeline.layout.teacher_selection.parent.mkdir(parents=True, exist_ok=True)
    pipeline.layout.teacher_selection.write_text(
        json.dumps({"selected_count": required, "selected": []})
    )

    with pytest.raises(RuntimeError, match="selected rows=0"):
        pipeline._validate_resumed_teacher_state(
            pipeline.layout.reports / "teacher_manifest.json"
        )


def test_conversion_manifest_is_bound_to_selection_and_sft_bytes(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(config, run_name="conversion-digest", dry_run=True)
    pipeline.layout.sft_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pipeline.layout.sft_jsonl.write_text('{"row":1}\n')
    selection = tmp_path / "selection.json"
    rollout_dir = tmp_path / "rollout"
    (rollout_dir / "trajectory").mkdir(parents=True)
    (rollout_dir / "result.json").write_text('{"reward":1}\n')
    trajectory = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    trajectory.write_text('{"exchange":1}\n')
    selection.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "task_id": "task-a",
                        "rollout_dir": str(rollout_dir),
                    }
                ]
            }
        )
    )
    source_digest = _teacher_sources_sha256(selection)
    conversion = pipeline.layout.reports / "sft_conversion.json"
    conversion.parent.mkdir(parents=True, exist_ok=True)
    conversion.write_text(
        json.dumps(
            {
                "teacher_selection_sha256": _sha256(selection),
                "teacher_sources_sha256": source_digest,
                "sft_jsonl_sha256": _sha256(pipeline.layout.sft_jsonl),
            }
        )
    )

    assert pipeline._conversion_matches_selection(
        conversion,
        selection_digest=_sha256(selection),
        source_digest=source_digest,
    )
    pipeline.layout.sft_jsonl.write_text('{"row":2}\n')
    assert not pipeline._conversion_matches_selection(
        conversion,
        selection_digest=_sha256(selection),
        source_digest=source_digest,
    )
    pipeline.layout.sft_jsonl.write_text('{"row":1}\n')
    trajectory.write_text('{"exchange":2}\n')
    assert not pipeline._conversion_matches_selection(
        conversion,
        selection_digest=_sha256(selection),
        source_digest=_teacher_sources_sha256(selection),
    )


def test_resume_checkpoint_digests_detect_tampering(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    pipeline = Pipeline(config, run_name="checkpoint-digests", dry_run=True)
    pipeline.layout.sft_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pipeline.layout.sft_jsonl.write_text('{"row":1}\n')
    for path, filename, content in (
        (pipeline.layout.sft_adapter, "adapter_model.safetensors", b"sft-adapter"),
        (pipeline.layout.sft_merged, "model.safetensors", b"sft-merged"),
        (pipeline.layout.grpo_adapter, "adapter_model.safetensors", b"grpo-adapter"),
        (pipeline.layout.grpo_merged, "model.safetensors", b"grpo-merged"),
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / filename).write_bytes(content)
    sft_metrics_path = pipeline.layout.sft_merged / "train_metrics.json"
    sft_metrics_path.write_text(
        json.dumps(
            {
                "mode": "sft",
                "base_model": config.model,
                "model_revision": config.model_revision,
                "adapter_dir": str(pipeline.layout.sft_adapter),
                "merged_model_dir": str(pipeline.layout.sft_merged),
                "train_jsonl_sha256": _sha256(pipeline.layout.sft_jsonl),
                "adapter_sha256": directory_sha256(pipeline.layout.sft_adapter),
                "merged_model_sha256": directory_sha256(pipeline.layout.sft_merged),
            }
        )
    )
    assert pipeline._sft_checkpoint_is_current(
        sft_metrics_path,
        output_model=pipeline.layout.sft_merged,
    )
    grpo_metrics_path = pipeline.layout.grpo_merged / "train_metrics.json"
    grpo_metrics_path.write_text(
        json.dumps(
            {
                "mode": "grpo",
                "model": str(pipeline.layout.sft_merged),
                "task_ids": pipeline.train_task_ids,
                "adapter_dir": str(pipeline.layout.grpo_adapter),
                "merged_model_dir": str(pipeline.layout.grpo_merged),
                "base_checkpoint_sha256": directory_sha256(pipeline.layout.sft_merged),
                "adapter_sha256": directory_sha256(pipeline.layout.grpo_adapter),
                "merged_model_sha256": directory_sha256(pipeline.layout.grpo_merged),
            }
        )
    )
    assert pipeline._grpo_checkpoint_is_current(
        grpo_metrics_path,
        input_model=str(pipeline.layout.sft_merged),
        output_model=pipeline.layout.grpo_merged,
    )

    (pipeline.layout.grpo_merged / "model.safetensors").write_bytes(b"tampered")
    assert not pipeline._grpo_checkpoint_is_current(
        grpo_metrics_path,
        input_model=str(pipeline.layout.sft_merged),
        output_model=pipeline.layout.grpo_merged,
    )


def test_resume_rejects_changed_run_plan_before_overwriting_it(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, output_root=tmp_path)
    original = Pipeline(config, run_name="resume-plan", dry_run=True)
    original._prepare_run_plan()
    original_plan = json.loads((original.layout.reports / "plan.json").read_text())
    changed = Pipeline(
        replace(
            config,
            teacher=replace(
                config.teacher, max_attempts=config.teacher.max_attempts + 1
            ),
        ),
        run_name="resume-plan",
        dry_run=True,
        resume=True,
    )

    with pytest.raises(RuntimeError, match="Changed fields: teacher"):
        changed._prepare_run_plan()

    assert json.loads((changed.layout.reports / "plan.json").read_text()) == (
        original_plan
    )


def test_resume_rejects_task_list_content_drift_with_the_same_count(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    train_list = tmp_path / "train.txt"
    train_ids = config.train_dataset.task_list.read_text().splitlines()
    train_list.write_text("\n".join(train_ids) + "\n")
    config = replace(
        config,
        output_root=tmp_path / "runs",
        train_dataset=replace(config.train_dataset, task_list=train_list),
    )
    original = Pipeline(config, run_name="resume-task-list", dry_run=True)
    original._prepare_run_plan()
    train_list.write_text("\n".join(reversed(train_ids)) + "\n")
    changed = Pipeline(
        config,
        run_name="resume-task-list",
        dry_run=True,
        resume=True,
    )

    with pytest.raises(RuntimeError, match="Changed fields: train_task_ids"):
        changed._prepare_run_plan()
