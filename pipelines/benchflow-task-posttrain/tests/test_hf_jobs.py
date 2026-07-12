from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

from posttrainarena.benchflow_pipeline.hf_jobs import (
    create_job_bundle,
    submit_hf_job,
)


ROOT = Path(__file__).resolve().parents[1]


def test_job_bundle_is_portable_and_contains_no_secrets(tmp_path: Path) -> None:
    benchmark_list = tmp_path / "benchmark.txt"
    benchmark_list.write_text("0000_369_369503_qa_1\n")
    benchmark_manifest = tmp_path / "benchmarks.toml"
    benchmark_manifest.write_text(
        "[[benchmarks]]\n"
        'name = "data-agent"\n'
        'repo_id = "benchflow/data_agent_rl_environment_eval"\n'
        'revision = "0ea976c79e3248c85737c4f7363484e4d47ce287"\n'
        'task_list = "benchmark.txt"\n'
    )
    bundle = create_job_bundle(
        config_path=ROOT / "configs/qwen3-4b-data-agent-forced-grpo-smoke.toml",
        output_dir=tmp_path / "bundle",
        run_id="run-1",
        submission_id="team-alpha",
        team_name="Team Alpha",
        artifact_repo="benchflow/results",
        leaderboard_repo="benchflow/leaderboard",
        model_repo="benchflow/model",
        benchmark_manifest=benchmark_manifest,
    )

    config = tomllib.loads(bundle.config_path.read_text())
    manifest = json.loads(bundle.manifest_path.read_text())
    text = "\n".join(
        path.read_text() for path in bundle.root.rglob("*") if path.is_file()
    )

    assert config["output"]["root"] == "runs"
    assert config["train_dataset"]["task_list"] == ("task-lists/train_dataset.txt")
    assert config["eval_dataset"]["task_list"] == ("task-lists/eval_dataset.txt")
    assert manifest["run_id"] == "run-1"
    assert manifest["benchmark_manifest"] == "benchmarks.toml"
    portable_benchmarks = tomllib.loads((bundle.root / "benchmarks.toml").read_text())
    assert portable_benchmarks["benchmarks"][0]["task_list"] == (
        "benchmark-task-lists/data-agent.txt"
    )
    assert "HF_TOKEN" not in text
    assert "DAYTONA_API_KEY" not in text


def test_submit_hf_job_passes_secrets_only_to_job_api(
    tmp_path: Path, monkeypatch
) -> None:
    import huggingface_hub

    bundle = create_job_bundle(
        config_path=ROOT / "configs/qwen3-4b-data-agent-forced-grpo-smoke.toml",
        output_dir=tmp_path / "bundle",
        run_id="run-1",
        submission_id="team-alpha",
        team_name="Team Alpha",
        artifact_repo="benchflow/results",
        leaderboard_repo="benchflow/leaderboard",
        model_repo=None,
    )
    calls: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def create_repo(self, *args, **kwargs):
            calls["create_repo"] = (args, kwargs)

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            return SimpleNamespace(oid="b" * 40)

    def fake_run_uv_job(script, **kwargs):
        calls["job"] = {"script": script, **kwargs}
        return SimpleNamespace(
            id="job-1",
            status=SimpleNamespace(stage="RUNNING"),
            url="https://huggingface.co/jobs/job-1",
            owner=SimpleNamespace(name="benchflow"),
        )

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "run_uv_job", fake_run_uv_job)
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.leaderboard.publish_record",
        lambda **kwargs: {"commit": "d" * 40},
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    result = submit_hf_job(
        bundle=bundle,
        artifact_repo="benchflow/results",
        posttrainarena_ref="c" * 40,
        flavor="cpu-basic",
        namespace="benchflow",
        timeout="10m",
        secret_names=["HF_TOKEN"],
        token="launcher-token",
        pipeline_dry_run=True,
    )

    assert result["job_id"] == "job-1"
    assert "secret-token" not in json.dumps(result)
    job_call = calls["job"]
    assert isinstance(job_call, dict)
    assert job_call["secrets"] == {"HF_TOKEN": "secret-token"}
    assert job_call["python"] == "3.12"
    assert "--pipeline-dry-run" in job_call["script_args"]
    assert os.path.basename(str(job_call["script"])) == "run_hf_job.py"
