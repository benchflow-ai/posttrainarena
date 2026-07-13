# /// script
# requires-python = "==3.12.*"
# dependencies = ["huggingface_hub>=0.36,<2"]
# ///
"""Run a pinned PostTrain Arena bundle on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-repo", required=True)
    parser.add_argument("--bundle-revision", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--posttrainarena-ref", required=True)
    parser.add_argument("--pipeline-dry-run", action="store_true")
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def install_pipeline(ref: str, *, train: bool) -> None:
    extras = "train,hf" if train else "hf"
    package = (
        f"posttrainarena-benchflow-pipeline[{extras}] @ "
        "git+https://github.com/benchflow-ai/posttrainarena.git"
        f"@{ref}#subdirectory=pipelines/benchflow-task-posttrain"
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--torch-backend",
            "auto",
            package,
        ],
        check=True,
    )


def restore_run_state(bundle: Path, manifest: dict[str, object]) -> bool:
    run_id = str(manifest["run_id"])
    run_dir = bundle / "runs" / run_id
    try:
        snapshot_download(
            str(manifest["artifact_repo"]),
            repo_type="dataset",
            allow_patterns=[f"runs/{run_id}/**"],
            local_dir=bundle,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        print(
            f"[posttrainarena] no resumable state restored: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    return run_dir.is_dir() and any(path.is_file() for path in run_dir.rglob("*"))


def main() -> int:
    args = parse_args()
    workspace = Path(tempfile.mkdtemp(prefix="posttrainarena-job-"))
    snapshot_download(
        args.bundle_repo,
        repo_type="dataset",
        revision=args.bundle_revision,
        allow_patterns=[f"{args.bundle_path}/**"],
        local_dir=workspace,
        token=os.environ.get("HF_TOKEN"),
    )
    bundle = workspace / args.bundle_path
    manifest = json.loads((bundle / "job.json").read_text())
    run_id = str(manifest["run_id"])
    run_dir = bundle / "runs" / run_id
    resume = restore_run_state(bundle, manifest)
    log_path = bundle / "job.log"
    install_pipeline(args.posttrainarena_ref, train=not args.pipeline_dry_run)
    executable = Path(sys.executable).parent / "posttrainarena-train"
    status_result = subprocess.run(
        [
            str(executable),
            "publish-status",
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--submission-id",
            str(manifest["submission_id"]),
            "--team-name",
            str(manifest["team_name"]),
            "--leaderboard-repo",
            str(manifest["leaderboard_repo"]),
            "--status",
            "running",
        ],
        check=False,
    )
    if status_result.returncode:
        print(
            "[posttrainarena] warning: failed to publish running status; "
            "continuing pipeline",
            flush=True,
        )
    command = [
        str(executable),
        "run",
        "--config",
        str(bundle / manifest["config"]),
        "--run-name",
        run_id,
    ]
    if args.pipeline_dry_run:
        command.append("--dry-run")
    elif resume:
        command.append("--resume")
    try:
        run_logged(command, log_path)
        if manifest.get("benchmark_manifest"):
            benchmark_command = [
                str(executable),
                "benchmarks",
                "--config",
                str(bundle / manifest["config"]),
                "--run-dir",
                str(run_dir),
                "--manifest",
                str(bundle / manifest["benchmark_manifest"]),
            ]
            if args.pipeline_dry_run:
                benchmark_command.append("--dry-run")
            run_logged(benchmark_command, log_path)
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, run_dir / "reports" / "hf_job.log")
        publish = [
            str(executable),
            "publish-run",
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--submission-id",
            str(manifest["submission_id"]),
            "--team-name",
            str(manifest["team_name"]),
            "--artifact-repo",
            str(manifest["artifact_repo"]),
            "--leaderboard-repo",
            str(manifest["leaderboard_repo"]),
        ]
        if manifest.get("model_repo") and not args.pipeline_dry_run:
            publish.extend(["--model-repo", str(manifest["model_repo"])])
            if manifest.get("private_model") is False:
                publish.append("--public-model")
        publish.append(
            "--public-artifacts"
            if manifest.get("private_artifacts") is False
            else "--private-artifacts"
        )
        run_logged(publish, log_path)
    except Exception as exc:
        (run_dir / "reports").mkdir(parents=True, exist_ok=True)
        if log_path.is_file():
            shutil.copy2(log_path, run_dir / "reports" / "hf_job.log")
        failure = [
            str(executable),
            "publish-failure",
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--submission-id",
            str(manifest["submission_id"]),
            "--team-name",
            str(manifest["team_name"]),
            "--artifact-repo",
            str(manifest["artifact_repo"]),
            "--leaderboard-repo",
            str(manifest["leaderboard_repo"]),
            "--error",
            f"{type(exc).__name__}: {exc}",
        ]
        failure.append(
            "--public-artifacts"
            if manifest.get("private_artifacts") is False
            else "--private-artifacts"
        )
        subprocess.run(failure, check=False)
        raise
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
