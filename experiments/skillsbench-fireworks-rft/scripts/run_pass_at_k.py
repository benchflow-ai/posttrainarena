from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from skillsbench_rft.roster import SOURCE_TASKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]


def build_command(*, k: int, concurrency: int, jobs_dir: Path) -> list[str]:
    command = [
        os.getenv("BENCHFLOW_BIN", "/home/cc2864/.local/bin/bench"),
        "eval",
        "run",
        "--tasks-dir",
        str(REPOSITORY_ROOT / "skillsbench" / "tasks"),
        "--matrix",
        str(PROJECT_ROOT / "eval-matrix.yaml"),
        "--trials",
        str(k),
        "--sandbox",
        "daytona",
        "--skill-mode",
        "no-skill",
        "--concurrency",
        str(concurrency),
        "--build-concurrency",
        str(min(concurrency, 8)),
        "--worker-concurrency",
        "4",
        "--worker-retries",
        "2",
        "--worker-start-stagger-sec",
        "2",
        "--usage-tracking",
        "required",
        "--expected-tasks",
        str(len(SOURCE_TASKS)),
        "--jobs-dir",
        str(jobs_dir),
        "--task-manifest-out",
        str(jobs_dir / "task-manifest.json"),
        "--run-config-out",
        str(jobs_dir / "run-config.json"),
        "--health-summary-out",
        str(jobs_dir / "health-summary.json"),
    ]
    for task_id in SOURCE_TASKS:
        command.extend(("--include", task_id))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eval-pass-at-10-20260909-v3",
    )
    args = parser.parse_args()

    fireworks_key = os.environ.get("FIREWORKS_API_KEY")
    if not fireworks_key:
        raise SystemExit("FIREWORKS_API_KEY is required")
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is required")

    env = os.environ.copy()
    env["BENCHFLOW_PROVIDER_BASE_URL"] = "https://api.fireworks.ai/inference/v1"
    env["BENCHFLOW_PROVIDER_API_KEY"] = fireworks_key
    env["OPENAI_API_KEY"] = fireworks_key
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_command(k=args.k, concurrency=args.concurrency, jobs_dir=args.jobs_dir),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
