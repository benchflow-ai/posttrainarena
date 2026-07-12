"""Evaluate one trained checkpoint across multiple pinned benchmark suites."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .io import CommandRunner, load_json, read_task_ids, write_json


SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    repo_id: str
    revision: str
    path: str
    task_list: Path
    weight: float = 1.0


def load_benchmark_manifest(path: Path) -> list[BenchmarkSuite]:
    source = path.expanduser().resolve()
    data = tomllib.loads(source.read_text())
    raw_suites = data.get("benchmarks")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ValueError("benchmark manifest requires [[benchmarks]] entries")
    suites: list[BenchmarkSuite] = []
    names: set[str] = set()
    for raw in raw_suites:
        name = str(raw["name"])
        if not SAFE_NAME.fullmatch(name):
            raise ValueError(f"invalid benchmark name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate benchmark name: {name}")
        names.add(name)
        task_list = Path(str(raw["task_list"])).expanduser()
        if not task_list.is_absolute():
            task_list = (source.parent / task_list).resolve()
        if not task_list.is_file():
            raise FileNotFoundError(task_list)
        weight = float(raw.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"benchmark {name} weight must be positive")
        suites.append(
            BenchmarkSuite(
                name=name,
                repo_id=str(raw["repo_id"]),
                revision=str(raw["revision"]),
                path=str(raw.get("path", "tasks")),
                task_list=task_list,
                weight=weight,
            )
        )
    return suites


def _snapshot_command(suite: BenchmarkSuite, destination: Path) -> list[str]:
    task_ids = read_task_ids(suite.task_list)
    command = [
        "bench",
        "tasks",
        "snapshot-hf",
        suite.repo_id,
        str(destination),
        "--revision",
        suite.revision,
        "--overwrite",
    ]
    if suite.path:
        command.extend(["--path", suite.path])
    command.extend(item for task_id in task_ids for item in ("--include-task", task_id))
    return command


def run_benchmark_matrix(
    *,
    config_path: Path,
    run_dir: Path,
    manifest_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    suites = load_benchmark_manifest(manifest_path)
    score_path = run_dir / "reports" / "score.json"
    if not score_path.is_file() and not dry_run:
        raise FileNotFoundError(f"Run score not found: {score_path}")
    score = load_json(score_path) if score_path.is_file() else {}
    final_model = str(score.get("final_model") or run_dir / "checkpoints" / "grpo")
    runner = CommandRunner(cwd=config.source.parent, dry_run=dry_run)
    results: list[dict[str, Any]] = []
    from .opencode import evaluate

    for suite in suites:
        task_ids = read_task_ids(suite.task_list)
        if not task_ids:
            raise ValueError(f"benchmark {suite.name} has no task IDs")
        tasks_dir = run_dir / "data" / "benchmarks" / suite.name
        runner.run(
            f"snapshot_benchmark_{suite.name}",
            _snapshot_command(suite, tasks_dir),
        )
        jobs_root = run_dir / "jobs" / "benchmarks" / suite.name
        results_root = run_dir / "results" / "benchmarks" / suite.name
        baseline = evaluate(
            config=config,
            runner=runner,
            stage=f"baseline_benchmark_{suite.name}",
            model=config.model,
            tasks_dir=tasks_dir,
            task_ids=task_ids,
            jobs_dir=jobs_root / "baseline",
            metrics_path=results_root / "baseline.json",
        )
        final = evaluate(
            config=config,
            runner=runner,
            stage=f"final_benchmark_{suite.name}",
            model=final_model,
            tasks_dir=tasks_dir,
            task_ids=task_ids,
            jobs_dir=jobs_root / "final",
            metrics_path=results_root / "final.json",
        )
        baseline_score = None if baseline["score"] is None else float(baseline["score"])
        final_score = None if final["score"] is None else float(final["score"])
        if not dry_run:
            runner.run(
                f"compare_benchmark_{suite.name}",
                [
                    "bench",
                    "eval",
                    "compare-lift",
                    "--baseline",
                    str(jobs_root / "baseline"),
                    "--trained",
                    str(jobs_root / "final"),
                    "--out",
                    str(results_root / "EVAL_LIFT.md"),
                    "--json-out",
                    str(results_root / "eval_lift.json"),
                ],
            )
        results.append(
            {
                "name": suite.name,
                "repo_id": suite.repo_id,
                "revision": suite.revision,
                "task_ids": task_ids,
                "task_count": len(task_ids),
                "weight": suite.weight,
                "baseline_score": baseline_score,
                "final_score": final_score,
                "delta_score": (
                    None
                    if baseline_score is None or final_score is None
                    else final_score - baseline_score
                ),
            }
        )
    scored = [item for item in results if item["delta_score"] is not None]
    weight_sum = sum(float(item["weight"]) for item in scored)
    macro_delta = (
        None
        if not scored
        else sum(float(item["delta_score"]) * float(item["weight"]) for item in scored)
        / weight_sum
    )
    payload = {
        "schema_version": 1,
        "run_name": run_dir.name,
        "base_model": config.model,
        "final_model": final_model,
        "benchmark_count": len(results),
        "macro_delta_score": macro_delta,
        "benchmarks": results,
        "commands": runner.commands,
    }
    write_json(run_dir / "reports" / "benchmarks" / "summary.json", payload)
    return payload
