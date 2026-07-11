"""Prepare portable job bundles and submit the pipeline through HF Jobs."""

from __future__ import annotations

import json
import os
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_json


TERMINAL_JOB_STAGES = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED"}


@dataclass(frozen=True)
class JobBundle:
    run_id: str
    root: Path
    config_path: Path
    manifest_path: Path


def _stage(value: Any) -> str:
    return str(getattr(value, "value", value))


def create_job_bundle(
    *,
    config_path: Path,
    output_dir: Path,
    run_id: str,
    submission_id: str,
    team_name: str,
    artifact_repo: str,
    leaderboard_repo: str,
    model_repo: str | None,
    benchmark_manifest: Path | None = None,
) -> JobBundle:
    import tomli_w

    source = config_path.expanduser().resolve()
    data = tomllib.loads(source.read_text())
    shutil.rmtree(output_dir, ignore_errors=True)
    task_lists = output_dir / "task-lists"
    task_lists.mkdir(parents=True)
    for table_name in ("train_dataset", "eval_dataset"):
        table = data[table_name]
        task_list = Path(str(table["task_list"])).expanduser()
        if not task_list.is_absolute():
            task_list = (source.parent / task_list).resolve()
        destination = task_lists / f"{table_name}.txt"
        shutil.copy2(task_list, destination)
        table["task_list"] = f"task-lists/{destination.name}"
    data.setdefault("output", {})["root"] = "runs"
    portable_config = output_dir / "config.toml"
    portable_config.write_text(tomli_w.dumps(data))
    portable_benchmarks: str | None = None
    if benchmark_manifest:
        benchmark_source = benchmark_manifest.expanduser().resolve()
        benchmark_data = tomllib.loads(benchmark_source.read_text())
        benchmark_lists = output_dir / "benchmark-task-lists"
        benchmark_lists.mkdir()
        for raw in benchmark_data.get("benchmarks", []):
            task_list = Path(str(raw["task_list"])).expanduser()
            if not task_list.is_absolute():
                task_list = (benchmark_source.parent / task_list).resolve()
            destination = benchmark_lists / f"{raw['name']}.txt"
            shutil.copy2(task_list, destination)
            raw["task_list"] = f"benchmark-task-lists/{destination.name}"
        benchmark_output = output_dir / "benchmarks.toml"
        benchmark_output.write_text(tomli_w.dumps(benchmark_data))
        portable_benchmarks = benchmark_output.name
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "submission_id": submission_id,
        "team_name": team_name,
        "config": "config.toml",
        "artifact_repo": artifact_repo,
        "leaderboard_repo": leaderboard_repo,
        "model_repo": model_repo,
        "benchmark_manifest": portable_benchmarks,
    }
    manifest_path = output_dir / "job.json"
    write_json(manifest_path, manifest)
    return JobBundle(run_id, output_dir, portable_config, manifest_path)


def submit_hf_job(
    *,
    bundle: JobBundle,
    artifact_repo: str,
    posttrainarena_ref: str,
    flavor: str,
    namespace: str | None,
    timeout: str,
    secret_names: list[str],
    token: str | None = None,
    pipeline_dry_run: bool = False,
    private_artifacts: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import HfApi, run_uv_job

    api = HfApi(token=token)
    api.create_repo(
        artifact_repo,
        repo_type="dataset",
        private=private_artifacts,
        exist_ok=True,
    )
    bundle_path = f"job-inputs/{bundle.run_id}"
    upload = api.upload_folder(
        repo_id=artifact_repo,
        repo_type="dataset",
        folder_path=bundle.root,
        path_in_repo=bundle_path,
        commit_message=f"Prepare HF Job {bundle.run_id}",
    )
    secrets: dict[str, str] = {}
    missing: list[str] = []
    for name in secret_names:
        value = os.environ.get(name)
        if value:
            secrets[name] = value
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing HF Job secrets: {', '.join(missing)}")
    script = Path(__file__).parent / "jobs" / "run_hf_job.py"
    args = [
        "--bundle-repo",
        artifact_repo,
        "--bundle-revision",
        upload.oid,
        "--bundle-path",
        bundle_path,
        "--posttrainarena-ref",
        posttrainarena_ref,
    ]
    if pipeline_dry_run:
        args.append("--pipeline-dry-run")
    job = run_uv_job(
        str(script),
        script_args=args,
        python="3.12",
        flavor=flavor,
        timeout=timeout,
        namespace=namespace,
        secrets=secrets,
        token=token,
    )
    result = {
        "run_id": bundle.run_id,
        "bundle_repo": artifact_repo,
        "bundle_revision": upload.oid,
        "bundle_path": bundle_path,
        "job_id": job.id,
        "job_status": _stage(job.status.stage),
        "job_url": job.url,
        "namespace": job.owner.name,
    }
    try:
        from .leaderboard import publish_record

        manifest = json.loads(bundle.manifest_path.read_text())
        result["leaderboard"] = publish_record(
            repo_id=str(manifest["leaderboard_repo"]),
            token=token,
            record={
                "run_id": bundle.run_id,
                "submission_id": str(manifest["submission_id"]),
                "team_name": str(manifest["team_name"]),
                "status": "queued",
                "job_id": job.id,
                "artifact_url": (
                    f"https://huggingface.co/datasets/{artifact_repo}"
                    f"/tree/main/runs/{bundle.run_id}"
                ),
            },
        )
    except Exception as exc:
        result["leaderboard_warning"] = f"{type(exc).__name__}: {exc}"
    return result


def job_status(
    *, job_id: str, namespace: str | None = None, token: str | None = None
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    job = HfApi(token=token).inspect_job(
        job_id=job_id,
        namespace=namespace,
    )
    return {
        "job_id": job.id,
        "status": _stage(job.status.stage),
        "message": job.status.message,
        "url": job.url,
        "flavor": str(job.flavor),
    }


def wait_for_job(
    *,
    job_id: str,
    namespace: str | None = None,
    token: str | None = None,
    timeout_seconds: int = 7200,
    poll_seconds: int = 15,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = job_status(job_id=job_id, namespace=namespace, token=token)
        if status["status"].upper() in TERMINAL_JOB_STAGES:
            return status
        time.sleep(poll_seconds)
    raise TimeoutError(f"HF Job {job_id} did not finish in {timeout_seconds}s")
