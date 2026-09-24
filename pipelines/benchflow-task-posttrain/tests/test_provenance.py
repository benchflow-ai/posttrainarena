from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import posttrainarena.benchflow_pipeline.provenance as provenance
from posttrainarena.benchflow_pipeline.config import BENCHFLOW_COMMIT, load_config
from posttrainarena.benchflow_pipeline.io import write_json
from posttrainarena.benchflow_pipeline.pipeline import Pipeline
from posttrainarena.benchflow_pipeline.provenance import recipe_identity


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"


def test_dry_run_writes_provenance(tmp_path: Path) -> None:
    config = replace(load_config(SMOKE), output_root=tmp_path)
    pipeline = Pipeline(config, run_name="prov", dry_run=True)

    pipeline.run()
    record = json.loads((pipeline.layout.reports / "provenance.json").read_text())

    assert record["schema_version"] == 1
    assert record["status"] == "complete"
    assert record["run_name"] == "prov"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    assert record["pipeline"]["commit"] == head
    assert record["pipeline"]["commit_source"] == "git"
    assert len(record["pipeline"]["package_source_sha256"]) == 64
    assert record["benchflow"]["pinned_commit"] == BENCHFLOW_COMMIT
    assert record["model"] == {"id": config.model, "revision": config.model_revision}
    assert len(record["recipe"]["recipe_sha256"]) == 64
    assert record["recipe"]["config_file_sha256"]
    assert record["sampler"] == {
        "task_sampler": "trl",
        "seed": 42,
        "report": "train_sampler.json",
    }
    train, heldout = record["datasets"]
    assert (train["role"], train["repo_id"], train["revision"]) == (
        "train",
        config.train_dataset.repo_id,
        config.train_dataset.revision,
    )
    assert (heldout["role"], heldout["name"], heldout["task_count"]) == ("eval", "eval", 2)
    # A dry run takes no snapshot, so there are no task digests.
    assert heldout["snapshot"] is None
    assert record["packages"]["trl"]


def test_provenance_carries_snapshot_task_digests(tmp_path: Path) -> None:
    config = replace(load_config(SMOKE), output_root=tmp_path)
    pipeline = Pipeline(config, run_name="digests", dry_run=False)
    write_json(
        pipeline.layout.reports / "train_snapshot_integrity.json",
        {
            "schema_version": 1,
            "resolved_revision": config.train_dataset.revision,
            "marker_sha256": "m" * 64,
            "task_sha256": {"task-a": "a" * 64},
            "reference_solutions_removed": ["task-a/oracle"],
        },
    )

    pipeline._write_provenance(status="running")
    record = json.loads((pipeline.layout.reports / "provenance.json").read_text())

    snapshot = record["datasets"][0]["snapshot"]
    assert snapshot["task_sha256"] == {"task-a": "a" * 64}
    assert snapshot["resolved_revision"] == config.train_dataset.revision
    assert snapshot["reference_solutions_removed"] == 1
    assert record["status"] == "running"


def test_recipe_hash_ignores_paths_but_tracks_the_recipe(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    moved_list = tmp_path / "train.txt"
    moved_list.write_text(config.train_dataset.task_list.read_text())
    moved = replace(
        config,
        output_root=tmp_path / "elsewhere",
        source=tmp_path / "copy.toml",
        train_dataset=replace(config.train_dataset, task_list=moved_list),
    )
    ids = {
        "train_task_ids": ["a", "b"],
        "suite_task_ids": {"eval": ["c"]},
    }

    base = recipe_identity(config, **ids)["recipe_sha256"]

    assert recipe_identity(moved, **ids)["recipe_sha256"] == base
    changed = replace(config, grpo=replace(config.grpo, learning_rate=2e-6))
    assert recipe_identity(changed, **ids)["recipe_sha256"] != base
    reseeded = replace(config, grpo=replace(config.grpo, seed=7))
    assert recipe_identity(reseeded, **ids)["recipe_sha256"] != base
    other_tasks = {**ids, "train_task_ids": ["a", "z"]}
    assert recipe_identity(config, **other_tasks)["recipe_sha256"] != base


def test_pipeline_commit_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "_git", lambda args, cwd: None)
    monkeypatch.setenv(provenance.PIPELINE_COMMIT_ENV, "abc123")

    source = provenance.pipeline_source()

    assert source["commit"] == "abc123"
    assert source["commit_source"] == "environment"
    assert source["dirty"] is None
