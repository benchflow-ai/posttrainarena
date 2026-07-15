from __future__ import annotations

import json
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.publishing import (
    build_run_record,
    publish_failure,
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
                "grpo_effective_update": True,
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
    assert record["grpo_effective_update"] is True
    assert record["train_task_count"] == 2
    assert record["eval_task_count"] == 1
    assert "contact_email" not in record


@pytest.mark.parametrize(
    "name",
    [
        "HF_TOKEN",
        "QWEN_API_KEY",
        "QWEN_BASE_URL",
        "OPENROUTER_API_KEY",
        "BENCHFLOW_MODEL_BRIDGE_CONTROL_URL",
    ],
)
def test_error_redaction_removes_known_secret_values(monkeypatch, name: str) -> None:
    monkeypatch.setenv(name, "very-secret-token")
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

        def update_repo_settings(self, *args, **kwargs):
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


def test_publish_run_uploads_final_model_and_lora_adapters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import huggingface_hub

    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    checkpoints = run_dir / "checkpoints"
    reports.mkdir(parents=True)
    for name in ("sft-adapter", "sft-merged", "grpo-adapter", "grpo-merged"):
        folder = checkpoints / name
        folder.mkdir(parents=True)
        (folder / "weights.safetensors").write_text(name)
    (reports / "score.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-9B",
                "final_model": str(checkpoints / "grpo-merged"),
                "baseline_score": 0.0,
                "score_after_posttrain": 0.5,
                "delta_score": 0.5,
                "train_task_ids": ["a"],
                "eval_task_ids": ["b"],
                "checkpoints": {
                    "sft_adapter": str(checkpoints / "sft-adapter"),
                    "sft_merged": str(checkpoints / "sft-merged"),
                    "grpo_adapter": str(checkpoints / "grpo-adapter"),
                    "grpo_merged": str(checkpoints / "grpo-merged"),
                },
            }
        )
    )
    uploads: list[dict[str, object]] = []
    creates: list[dict[str, object]] = []
    settings: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *args, **kwargs):
            creates.append({"args": args, **kwargs})
            return None

        def update_repo_settings(self, *args, **kwargs):
            settings.append({"args": args, **kwargs})

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)
            return type(
                "Commit",
                (),
                {
                    "oid": f"{len(uploads):040d}",
                    "pr_url": (
                        f"https://example.test/pr/{len(uploads)}"
                        if kwargs.get("create_pr")
                        else None
                    ),
                    "commit_url": f"https://example.test/{len(uploads)}",
                },
            )()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.publishing.publish_record",
        lambda **kwargs: {"commit": "b" * 40},
    )

    result = publish_run(
        run_dir=run_dir,
        run_id="trained",
        submission_id="submission",
        team_name="Team",
        artifact_repo="org/artifacts",
        leaderboard_repo="org/leaderboard",
        model_repo="org/model",
        model_create_pr=True,
    )

    model_uploads = [call for call in uploads if call.get("repo_type") == "model"]
    assert {call["path_in_repo"] for call in model_uploads} == {
        "runs/trained/final-merged",
        "runs/trained/sft-adapter",
        "runs/trained/sft-merged",
        "runs/trained/grpo-adapter",
    }
    assert set(result["model"]["artifacts"]) == {
        "final-merged",
        "sft-adapter",
        "sft-merged",
        "grpo-adapter",
    }
    assert result["model"]["commit"] == "0000000000000000000000000000000000000001"
    assert result["model"]["url"] == "https://example.test/pr/1"
    model_create = next(call for call in creates if call.get("repo_type") == "model")
    assert model_create["private"] is True
    assert settings == [
        {
            "args": ("org/model",),
            "repo_type": "model",
            "private": True,
        },
        {
            "args": ("org/artifacts",),
            "repo_type": "dataset",
            "private": True,
        },
    ]


def test_publish_run_rejects_checkpoint_path_outside_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import huggingface_hub

    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (reports / "score.json").write_text(
        json.dumps(
            {
                "model": "model",
                "final_model": str(outside),
                "baseline_score": 0.0,
                "score_after_posttrain": 0.0,
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

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    with pytest.raises(RuntimeError, match="outside"):
        publish_run(
            run_dir=run_dir,
            run_id="unsafe",
            submission_id="submission",
            team_name="Team",
            artifact_repo="org/artifacts",
            leaderboard_repo="org/leaderboard",
            model_repo="org/model",
        )


def test_publish_run_rejects_missing_declared_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import huggingface_hub

    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    final_model = run_dir / "checkpoints" / "grpo-merged"
    reports.mkdir(parents=True)
    final_model.mkdir(parents=True)
    (reports / "score.json").write_text(
        json.dumps(
            {
                "model": "model",
                "final_model": str(final_model),
                "baseline_score": 0.0,
                "score_after_posttrain": 0.0,
                "train_task_ids": ["a"],
                "eval_task_ids": ["b"],
                "checkpoints": {
                    "grpo_adapter": str(
                        run_dir / "checkpoints" / "missing-grpo-adapter"
                    )
                },
            }
        )
    )

    class FakeApi:
        def __init__(self, token=None):
            pass

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    with pytest.raises(FileNotFoundError, match="missing-grpo-adapter"):
        publish_run(
            run_dir=run_dir,
            run_id="missing-adapter",
            submission_id="submission",
            team_name="Team",
            artifact_repo="org/artifacts",
            leaderboard_repo="org/leaderboard",
            model_repo="org/model",
        )


def test_publish_failure_checkpoints_resumable_private_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import huggingface_hub

    run_dir = tmp_path / "run"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "jobs" / "teacher").mkdir(parents=True)
    (run_dir / "jobs" / "teacher" / "results.jsonl").write_text("{}\n")
    calls: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *args, **kwargs):
            calls["create"] = {"args": args, **kwargs}

        def update_repo_settings(self, *args, **kwargs):
            calls["settings"] = {"args": args, **kwargs}

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            return type("Commit", (), {"oid": "a" * 40})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.publishing.publish_record",
        lambda **kwargs: {"commit": "b" * 40},
    )

    result = publish_failure(
        run_dir=run_dir,
        run_id="failed-run",
        submission_id="submission",
        team_name="Team",
        artifact_repo="org/artifacts",
        leaderboard_repo="org/leaderboard",
        error="boom",
        private_artifacts=True,
    )

    assert result["resume_available"] is True
    assert calls["create"] == {
        "args": ("org/artifacts",),
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    upload = calls["upload"]
    assert isinstance(upload, dict)
    assert upload["folder_path"] == run_dir
    assert upload["path_in_repo"] == "runs/failed-run"
