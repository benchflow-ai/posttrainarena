"""Command-line interface for the public BenchFlow pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline


TEACHER_PROVIDER_SECRETS = {
    "glm": ("GLM_API_KEY", "GLM_BASE_URL"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "qwen-dashscope": ("QWEN_API_KEY", "QWEN_BASE_URL"),
}


def default_hf_job_secrets(config_path: Path) -> list[str]:
    config = load_config(config_path)
    names = ["HF_TOKEN"]
    if config.sandbox == "daytona":
        names.append("DAYTONA_API_KEY")
    if config.teacher.enabled:
        provider = config.teacher.model.split("/", 1)[0]
        provider_secrets = TEACHER_PROVIDER_SECRETS.get(provider)
        if provider_secrets is None:
            raise ValueError(
                "No default HF Job secret mapping for teacher provider "
                f"{provider!r}; pass --secret-env explicitly"
            )
        names.extend(provider_secrets)
    names.extend(
        [
            "BENCHFLOW_BASE_MODEL",
            "BENCHFLOW_ADAPTER_MODEL",
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
            "TRL_VLLM_SERVER_BASE_URL",
        ]
    )
    if config.tracking.report_to != "none":
        names.append("WANDB_API_KEY")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="posttrainarena-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"plan", "run"}:
            command.add_argument("--run-name")
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--resume", action="store_true")
    prepare = subparsers.add_parser("prepare-submission")
    prepare.add_argument("--entry", type=Path, required=True)
    prepare.add_argument("--base-config", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--dataset-repo", required=True)
    prepare.add_argument("--dataset-revision")
    prepare.add_argument("--upload", action="store_true")
    visibility = prepare.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    prepare.set_defaults(private=True)

    submit = subparsers.add_parser("hf-job-submit")
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--bundle-dir", type=Path, required=True)
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--submission-id", required=True)
    submit.add_argument("--team-name", required=True)
    submit.add_argument("--artifact-repo", required=True)
    submit.add_argument("--leaderboard-repo", required=True)
    submit.add_argument("--model-repo")
    submit.add_argument("--public-model", action="store_true")
    submit.add_argument("--benchmarks", type=Path)
    submit.add_argument("--posttrainarena-ref")
    submit.add_argument("--flavor", default="h100")
    submit.add_argument("--namespace")
    submit.add_argument("--timeout", default="6h")
    submit.add_argument("--secret-env", action="append")
    submit.add_argument("--pipeline-dry-run", action="store_true")
    submit.add_argument("--launcher-dry-run", action="store_true")
    artifact_visibility = submit.add_mutually_exclusive_group()
    artifact_visibility.add_argument(
        "--private-artifacts",
        dest="private_artifacts",
        action="store_true",
    )
    artifact_visibility.add_argument(
        "--public-artifacts",
        dest="private_artifacts",
        action="store_false",
    )
    submit.set_defaults(private_artifacts=True)
    submit.add_argument("--wait", action="store_true")

    status = subparsers.add_parser("hf-job-status")
    status.add_argument("--job-id", required=True)
    status.add_argument("--namespace")

    publish = subparsers.add_parser("publish-run")
    _add_publish_args(publish)
    publish.add_argument("--model-repo")
    publish.add_argument("--public-model", action="store_true")
    publish.add_argument("--model-create-pr", action="store_true")
    publish_visibility = publish.add_mutually_exclusive_group()
    publish_visibility.add_argument(
        "--private-artifacts",
        dest="private_artifacts",
        action="store_true",
    )
    publish_visibility.add_argument(
        "--public-artifacts",
        dest="private_artifacts",
        action="store_false",
    )
    publish.set_defaults(private_artifacts=True)

    failure = subparsers.add_parser("publish-failure")
    _add_publish_args(failure)
    failure.add_argument("--error", required=True)
    failure_visibility = failure.add_mutually_exclusive_group()
    failure_visibility.add_argument(
        "--private-artifacts",
        dest="private_artifacts",
        action="store_true",
    )
    failure_visibility.add_argument(
        "--public-artifacts",
        dest="private_artifacts",
        action="store_false",
    )
    failure.set_defaults(private_artifacts=True)

    publish_status = subparsers.add_parser("publish-status")
    publish_status.add_argument("--run-dir", type=Path, required=True)
    publish_status.add_argument("--run-id", required=True)
    publish_status.add_argument("--submission-id", required=True)
    publish_status.add_argument("--team-name", required=True)
    publish_status.add_argument("--leaderboard-repo", required=True)
    publish_status.add_argument("--job-id")
    publish_status.add_argument(
        "--status", choices=("queued", "running"), required=True
    )

    deploy = subparsers.add_parser("deploy-leaderboard")
    deploy.add_argument("--space-repo", required=True)
    deploy.add_argument("--leaderboard-repo", required=True)
    deploy.add_argument("--private", action="store_true")

    benchmarks = subparsers.add_parser("benchmarks")
    benchmarks.add_argument("--config", type=Path, required=True)
    benchmarks.add_argument("--run-dir", type=Path, required=True)
    benchmarks.add_argument("--manifest", type=Path, required=True)
    benchmarks.add_argument("--dry-run", action="store_true")

    serve = subparsers.add_parser("openenv-serve")
    serve.add_argument("--tasks-dir", type=Path, required=True)
    serve.add_argument("--include-task", action="append", default=[])
    serve.add_argument("--environment", choices=("docker", "daytona"), default="docker")
    serve.add_argument("--sandbox-user", default="agent")
    serve.add_argument("--jobs-dir", type=Path, default=Path("jobs/openenv"))
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    bridge = subparsers.add_parser("model-bridge")
    bridge.add_argument("--upstream-url")
    bridge.add_argument("--tokenizer", required=True)
    bridge.add_argument("--tokenizer-revision")
    bridge.add_argument("--api-key-env", default="BENCHFLOW_PROVIDER_API_KEY")
    bridge.add_argument("--max-tokens", type=int, default=4096)
    bridge.add_argument("--host", default="0.0.0.0")
    bridge.add_argument("--port", type=int, default=8001)
    return parser


def _add_publish_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--artifact-repo", required=True)
    parser.add_argument("--leaderboard-repo", required=True)
    parser.add_argument("--job-id")


def _git_ref() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if args.command == "prepare-submission":
        from .submission import prepare_submission

        prepared = prepare_submission(
            entry_dir=args.entry,
            base_config_path=args.base_config,
            output_dir=args.out,
            dataset_repo=args.dataset_repo,
            dataset_revision=args.dataset_revision,
            upload=args.upload,
            private=args.private,
            token=token,
        )
        print(json.dumps(prepared.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "hf-job-submit":
        from .hf_jobs import create_job_bundle, submit_hf_job, wait_for_job

        bundle = create_job_bundle(
            config_path=args.config,
            output_dir=args.bundle_dir,
            run_id=args.run_id,
            submission_id=args.submission_id,
            team_name=args.team_name,
            artifact_repo=args.artifact_repo,
            leaderboard_repo=args.leaderboard_repo,
            model_repo=args.model_repo,
            private_model=not args.public_model,
            private_artifacts=args.private_artifacts,
            benchmark_manifest=args.benchmarks,
        )
        secret_names = args.secret_env or (
            ["HF_TOKEN"]
            if args.pipeline_dry_run
            else default_hf_job_secrets(args.config)
        )
        resolved_ref = args.posttrainarena_ref or _git_ref()
        if args.launcher_dry_run:
            print(
                json.dumps(
                    {
                        "run_id": bundle.run_id,
                        "bundle_dir": str(bundle.root),
                        "posttrainarena_ref": resolved_ref,
                        "flavor": args.flavor,
                        "namespace": args.namespace,
                        "timeout": args.timeout,
                        "secret_names": secret_names,
                        "pipeline_dry_run": args.pipeline_dry_run,
                        "submitted": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = submit_hf_job(
            bundle=bundle,
            artifact_repo=args.artifact_repo,
            posttrainarena_ref=resolved_ref,
            flavor=args.flavor,
            namespace=args.namespace,
            timeout=args.timeout,
            secret_names=secret_names,
            token=token,
            pipeline_dry_run=args.pipeline_dry_run,
            private_artifacts=args.private_artifacts,
        )
        if args.wait:
            result["terminal"] = wait_for_job(
                job_id=result["job_id"],
                namespace=args.namespace,
                token=token,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "hf-job-status":
        from .hf_jobs import job_status

        print(
            json.dumps(
                job_status(
                    job_id=args.job_id,
                    namespace=args.namespace,
                    token=token,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "publish-run":
        from .publishing import publish_run

        result = publish_run(
            run_dir=args.run_dir,
            run_id=args.run_id,
            submission_id=args.submission_id,
            team_name=args.team_name,
            artifact_repo=args.artifact_repo,
            leaderboard_repo=args.leaderboard_repo,
            model_repo=args.model_repo,
            job_id=args.job_id,
            token=token,
            private_artifacts=args.private_artifacts,
            private_model=not args.public_model,
            model_create_pr=args.model_create_pr,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "publish-failure":
        from .publishing import publish_failure

        result = publish_failure(
            run_dir=args.run_dir,
            run_id=args.run_id,
            submission_id=args.submission_id,
            team_name=args.team_name,
            artifact_repo=args.artifact_repo,
            leaderboard_repo=args.leaderboard_repo,
            error=args.error,
            job_id=args.job_id,
            token=token,
            private_artifacts=args.private_artifacts,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "publish-status":
        from .publishing import publish_status

        result = publish_status(
            run_dir=args.run_dir,
            run_id=args.run_id,
            submission_id=args.submission_id,
            team_name=args.team_name,
            leaderboard_repo=args.leaderboard_repo,
            status=args.status,
            job_id=args.job_id,
            token=token,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "deploy-leaderboard":
        from .leaderboard import deploy_space

        result = deploy_space(
            space_repo=args.space_repo,
            leaderboard_repo=args.leaderboard_repo,
            token=token,
            private=args.private,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmarks":
        from .benchmarks import run_benchmark_matrix

        result = run_benchmark_matrix(
            config_path=args.config,
            run_dir=args.run_dir,
            manifest_path=args.manifest,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "openenv-serve":
        from .openenv_service import serve_openenv

        serve_openenv(
            tasks_dir=args.tasks_dir,
            include_tasks=args.include_task,
            environment=args.environment,
            sandbox_user=args.sandbox_user,
            jobs_dir=args.jobs_dir,
            host=args.host,
            port=args.port,
        )
        return 0
    if args.command == "model-bridge":
        from .model_bridge import serve_model_bridge

        upstream_url = args.upstream_url or os.environ.get("TRL_VLLM_SERVER_BASE_URL")
        if not upstream_url:
            raise RuntimeError(
                "model-bridge requires --upstream-url or TRL_VLLM_SERVER_BASE_URL"
            )
        serve_model_bridge(
            upstream_url=upstream_url,
            tokenizer_id=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            api_key=os.environ.get(args.api_key_env),
            max_tokens_per_call=args.max_tokens,
            host=args.host,
            port=args.port,
        )
        return 0
    config = load_config(args.config)
    if args.command == "validate":
        print(
            json.dumps(
                {"config": str(config.source), "valid": True},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    pipeline = Pipeline(
        config,
        run_name=args.run_name,
        dry_run=getattr(args, "dry_run", False),
        resume=getattr(args, "resume", False),
    )
    if args.command == "plan":
        print(json.dumps(pipeline.plan(), indent=2, sort_keys=True, default=str))
        return 0
    result = pipeline.run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
