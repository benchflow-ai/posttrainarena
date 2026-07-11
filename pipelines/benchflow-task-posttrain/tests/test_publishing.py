from __future__ import annotations

import json
from pathlib import Path

from posttrainarena.benchflow_pipeline.publishing import (
    build_run_record,
    publish_run,
    redact_error,
)


def test_build_run_record_reads_score_and_removes_contact_data(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "score.json").write_text(
        json.dumps(
            {
                "model": "model",
                "model_revision": "a" * 40,
                "final_model": "/tmp/final",
                "baseline_score": 0.1,
                "score_after_posttrain": 0.25,
                "delta_score": 0.15,
                "grpo_ran": True,
                "train_task_ids": ["a", "b"],
                "eval_task_ids": ["c"],
            }
        )
    )

    record = build_run_record(
        run_dir=run_dir,
        run_id="run-1",
        submission_id="team-alpha",
        team_name="Team Alpha",
        status="succeeded",
    )

    assert record["delta_score"] == 0.15
    assert record["train_task_count"] == 2
    assert record["eval_task_count"] == 1
    assert "contact_email" not in record


def test_error_redaction_removes_known_secret_values(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "very-secret-token")
    assert "very-secret-token" not in redact_error(
        "request failed for very-secret-token"
    )


def test_publish_run_marks_empty_score_as_dry_run(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "score.json").write_text(
        json.dumps(
            {
                "model": "model",
                "baseline_score": None,
                "score_after_posttrain": None,
                "delta_score": None,
                "train_task_ids": ["a"],
                "eval_task_ids": ["b"],
            }
        )
    )

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *args, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            return type("Commit", (), {"oid": "a" * 40})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.publishing.publish_record",
        lambda **kwargs: {"commit": "b" * 40},
    )

    result = publish_run(
        run_dir=run_dir,
        run_id="dry-run",
        submission_id="submission",
        team_name="Team",
        artifact_repo="org/artifacts",
        leaderboard_repo="org/leaderboard",
    )

    assert result["run"]["status"] == "dry-run"
