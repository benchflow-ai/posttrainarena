from __future__ import annotations

import json
import tomllib
from pathlib import Path

from posttrainarena.benchflow_pipeline.submission import prepare_submission


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_submission_emits_pinned_portable_recipe(tmp_path: Path) -> None:
    entry = tmp_path / "team-alpha"
    task = entry / "envs" / "task-one"
    task.mkdir(parents=True)
    (entry / "submission.yaml").write_text(
        "team_name: Team Alpha\ncontact_email: alpha@example.com\ntrack: environments\n"
    )
    (task / "task.md").write_text("---\nversion: '1.0'\n---\n\n## prompt\nSolve.\n")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text("#!/bin/sh\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/sh\n")
    prepared = prepare_submission(
        entry_dir=entry,
        base_config_path=ROOT / "configs/qwen3-4b-data-agent-smoke.toml",
        output_dir=tmp_path / "prepared",
        dataset_repo="benchflow/posttrainarena-team-alpha",
        dataset_revision="a" * 40,
    )

    recipe = tomllib.loads(prepared.recipe_path.read_text())
    manifest = json.loads(prepared.manifest_path.read_text())
    dataset_manifest = json.loads(
        (tmp_path / "prepared/dataset/submission.json").read_text()
    )

    assert prepared.submission_id == "team-alpha"
    assert prepared.task_count == 1
    assert recipe["train_dataset"]["repo_id"] == ("benchflow/posttrainarena-team-alpha")
    assert recipe["train_dataset"]["revision"] == "a" * 40
    assert recipe["train_dataset"]["task_list"] == "task-lists/train.txt"
    assert recipe["eval_dataset"]["task_list"] == "task-lists/eval.txt"
    assert recipe["teacher"]["min_verified"] == 1
    assert recipe["teacher"]["require_all_tasks"] is True
    assert manifest["task_count"] == 1
    assert dataset_manifest["source_entry"] == "team-alpha"
    assert "alpha@example.com" not in json.dumps(dataset_manifest)


def test_prepare_submission_rejects_incomplete_task(tmp_path: Path) -> None:
    entry = tmp_path / "team-alpha"
    task = entry / "envs" / "broken"
    task.mkdir(parents=True)
    (entry / "submission.yaml").write_text(
        "team_name: Team Alpha\ncontact_email: alpha@example.com\ntrack: environments\n"
    )
    (task / "task.md").write_text("---\nversion: '1.0'\n---\n\n## prompt\nSolve.\n")

    try:
        prepare_submission(
            entry_dir=entry,
            base_config_path=ROOT / "configs/qwen3-4b-data-agent-smoke.toml",
            output_dir=tmp_path / "prepared",
            dataset_repo="benchflow/posttrainarena-team-alpha",
            dataset_revision="a" * 40,
        )
    except ValueError as exc:
        assert "missing environment/Dockerfile" in str(exc)
    else:
        raise AssertionError("expected ValueError")
