"""Serve pinned BenchFlow tasks through the OpenEnv protocol."""

from __future__ import annotations

from pathlib import Path


def serve_openenv(
    *,
    tasks_dir: Path,
    include_tasks: list[str],
    environment: str,
    sandbox_user: str | None,
    jobs_dir: Path,
    host: str,
    port: int,
    bash_timeout_sec: int = 120,
    max_output_chars: int = 8192,
) -> None:
    from benchflow.integrations.trl import BashHarnessConfig, BenchFlowSpec
    import uvicorn

    from .openenv.server import create_openenv_app

    spec = BenchFlowSpec(
        tasks_dir=tasks_dir,
        include_tasks=include_tasks,
        bash_harness=BashHarnessConfig(
            environment=environment,
            sandbox_user=sandbox_user,
            jobs_dir=jobs_dir,
            reset_message=(
                "Use run_bash to inspect and solve the task. "
                "Call submit with only the final answer."
            ),
            bash_timeout_sec=bash_timeout_sec,
            max_output_chars=max_output_chars,
        ),
    )
    rows = {
        str(row["benchflow_task_id"]): dict(row)
        for row in spec.train_dataset_rows
    }
    app = create_openenv_app(spec.environment_factory, rows)
    uvicorn.run(app, host=host, port=port)
