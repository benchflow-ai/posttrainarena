"""Publish completed run artifacts, checkpoints, and leaderboard records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .leaderboard import publish_record, utc_now


SECRET_ENV_NAMES = (
    "HF_TOKEN",
    "DAYTONA_API_KEY",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    "OPENROUTER_API_KEY",
    "BENCHFLOW_BASE_MODEL",
    "BENCHFLOW_ADAPTER_MODEL",
    "BENCHFLOW_PROVIDER_BASE_URL",
    "BENCHFLOW_MODEL_BRIDGE_CONTROL_URL",
    "BENCHFLOW_PROVIDER_API_KEY",
    "TRL_VLLM_SERVER_BASE_URL",
    "WANDB_API_KEY",
)


def redact_error(value: str) -> str:
    redacted = value
    for name in SECRET_ENV_NAMES:
        secret = os.environ.get(name)
        if secret:
            redacted = redacted.replace(secret, f"<{name}:redacted>")
    return redacted[:2000]


def build_run_record(
    *,
    run_dir: Path,
    run_id: str,
    submission_id: str,
    team_name: str,
    status: str,
    job_id: str | None = None,
    artifact_url: str | None = None,
    model_url: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    score_path = run_dir / "reports" / "score.json"
    score = load_json(score_path) if score_path.is_file() else {}
    benchmark_path = run_dir / "reports" / "benchmarks" / "summary.json"
    benchmark_summary = load_json(benchmark_path) if benchmark_path.is_file() else None
    return {
        "schema_version": 1,
        "run_id": run_id,
        "submission_id": submission_id,
        "team_name": team_name,
        "status": status,
        "job_id": job_id,
        "model": score.get("model"),
        "model_revision": score.get("model_revision"),
        "final_model": score.get("final_model"),
        "baseline_score": score.get("baseline_score"),
        "score_after_posttrain": score.get("score_after_posttrain"),
        "delta_score": score.get("delta_score"),
        "macro_benchmark_delta": (
            benchmark_summary.get("macro_delta_score") if benchmark_summary else None
        ),
        "benchmark_scores": (
            benchmark_summary.get("benchmarks") if benchmark_summary else []
        ),
        "grpo_ran": score.get("grpo_ran"),
        "grpo_effective_update": score.get("grpo_effective_update"),
        "train_task_count": len(score.get("train_task_ids", [])),
        "eval_task_count": len(score.get("eval_task_ids", [])),
        "artifact_url": artifact_url,
        "model_url": model_url,
        "error": error,
        "updated_at": utc_now(),
    }


def publish_run(
    *,
    run_dir: Path,
    run_id: str,
    submission_id: str,
    team_name: str,
    artifact_repo: str,
    leaderboard_repo: str,
    model_repo: str | None = None,
    job_id: str | None = None,
    token: str | None = None,
    private_artifacts: bool = True,
    private_model: bool = True,
    model_create_pr: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    score = load_json(run_dir / "reports" / "score.json")
    api = HfApi(token=token)
    model_info: dict[str, Any] | None = None
    final_model = Path(str(score.get("final_model", ""))).expanduser()
    if model_repo:
        if not final_model.is_dir():
            raise FileNotFoundError(f"Final model checkpoint not found: {final_model}")
        checkpoint_root = (run_dir / "checkpoints").resolve()
        checkpoint_paths = score.get("checkpoints")
        candidates = [("final-merged", final_model)]
        if isinstance(checkpoint_paths, dict):
            for label in ("sft_adapter", "sft_merged", "grpo_adapter"):
                value = checkpoint_paths.get(label)
                if value:
                    candidates.append((label.replace("_", "-"), Path(str(value))))
        safe_candidates: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for label, folder in candidates:
            resolved = folder.expanduser().resolve()
            if not resolved.is_relative_to(checkpoint_root):
                raise RuntimeError(
                    f"Refusing to publish checkpoint outside {checkpoint_root}: "
                    f"{resolved}"
                )
            if not resolved.is_dir():
                raise FileNotFoundError(f"Declared checkpoint not found: {resolved}")
            if resolved in seen:
                continue
            seen.add(resolved)
            safe_candidates.append((label, resolved))
        if not safe_candidates:
            raise RuntimeError("No model checkpoints were available to publish")
        api.create_repo(
            model_repo,
            repo_type="model",
            private=private_model,
            exist_ok=True,
        )
        api.update_repo_settings(
            model_repo,
            repo_type="model",
            private=private_model,
        )
        uploads: dict[str, dict[str, str | None]] = {}
        for label, resolved in safe_candidates:
            commit = api.upload_folder(
                repo_id=model_repo,
                repo_type="model",
                folder_path=resolved,
                path_in_repo=f"runs/{run_id}/{label}",
                ignore_patterns=["checkpoint-*", "optimizer.pt", "completions/**"],
                commit_message=f"Publish PostTrain Arena {label} {run_id}",
                create_pr=model_create_pr,
            )
            uploads[label] = {
                "commit": commit.oid,
                "url": getattr(commit, "pr_url", None)
                or getattr(commit, "commit_url", None),
            }
        final_upload = uploads["final-merged"]
        model_url = (
            final_upload["url"]
            if model_create_pr and final_upload["url"]
            else f"https://huggingface.co/{model_repo}/tree/main/runs/{run_id}"
        )
        model_info = {
            "repo_id": model_repo,
            "commit": final_upload["commit"],
            "url": model_url,
            "artifacts": uploads,
        }
    api.create_repo(
        artifact_repo,
        repo_type="dataset",
        private=private_artifacts,
        exist_ok=True,
    )
    api.update_repo_settings(
        artifact_repo,
        repo_type="dataset",
        private=private_artifacts,
    )
    expected_artifact_url = (
        f"https://huggingface.co/datasets/{artifact_repo}/tree/main/runs/{run_id}"
    )
    status = (
        "dry-run"
        if score.get("baseline_score") is None
        and score.get("score_after_posttrain") is None
        else "succeeded"
    )
    record = build_run_record(
        run_dir=run_dir,
        run_id=run_id,
        submission_id=submission_id,
        team_name=team_name,
        status=status,
        job_id=job_id,
        artifact_url=expected_artifact_url,
        model_url=model_info["url"] if model_info else None,
    )
    write_json(run_dir / "reports" / "hub_run.json", record)
    artifact_commit = api.upload_folder(
        repo_id=artifact_repo,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=f"runs/{run_id}",
        ignore_patterns=[
            "checkpoints/**",
            "data/train/**",
            "data/eval/**",
        ],
        commit_message=f"Publish PostTrain Arena run {run_id}",
    )
    leaderboard = publish_record(
        repo_id=leaderboard_repo,
        record=record,
        token=token,
    )
    return {
        "run": record,
        "artifact_commit": artifact_commit.oid,
        "artifact_url": expected_artifact_url,
        "model": model_info,
        "leaderboard": leaderboard,
    }


def publish_failure(
    *,
    run_dir: Path,
    run_id: str,
    submission_id: str,
    team_name: str,
    artifact_repo: str,
    leaderboard_repo: str,
    error: str,
    job_id: str | None = None,
    token: str | None = None,
    private_artifacts: bool = True,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_url = (
        f"https://huggingface.co/datasets/{artifact_repo}/tree/main/runs/{run_id}"
    )
    record = build_run_record(
        run_dir=run_dir,
        run_id=run_id,
        submission_id=submission_id,
        team_name=team_name,
        status="failed",
        job_id=job_id,
        artifact_url=artifact_url,
        error=redact_error(error),
    )
    write_json(run_dir / "reports" / "hub_run.json", record)
    api = HfApi(token=token)
    api.create_repo(
        artifact_repo,
        repo_type="dataset",
        private=private_artifacts,
        exist_ok=True,
    )
    api.update_repo_settings(
        artifact_repo,
        repo_type="dataset",
        private=private_artifacts,
    )
    commit = api.upload_folder(
        repo_id=artifact_repo,
        repo_type="dataset",
        folder_path=run_dir,
        path_in_repo=f"runs/{run_id}",
        ignore_patterns=[
            "data/train/**",
            "data/eval/**",
        ],
        commit_message=f"Checkpoint failed PostTrain Arena run {run_id}",
    )
    leaderboard = publish_record(
        repo_id=leaderboard_repo,
        record=record,
        token=token,
    )
    return {
        "run": record,
        "artifact_commit": commit.oid,
        "artifact_url": artifact_url,
        "resume_available": any(
            path.is_file()
            for root in (run_dir / "jobs", run_dir / "checkpoints")
            if root.is_dir()
            for path in root.rglob("*")
        ),
        "leaderboard": leaderboard,
    }


def publish_status(
    *,
    run_dir: Path,
    run_id: str,
    submission_id: str,
    team_name: str,
    leaderboard_repo: str,
    status: str,
    job_id: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    if status not in {"queued", "running"}:
        raise ValueError("publish_status supports queued or running")
    record = build_run_record(
        run_dir=run_dir,
        run_id=run_id,
        submission_id=submission_id,
        team_name=team_name,
        status=status,
        job_id=job_id,
    )
    return publish_record(
        repo_id=leaderboard_repo,
        record=record,
        token=token,
    )
