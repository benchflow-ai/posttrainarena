"""Run provenance: which code, model, recipe and task bytes produced a run.

`reports/provenance.json` is written once the task snapshots exist (so a run that later fails
still records what it ran on) and rewritten at the end with the checkpoint digests. It follows
the MiMo `write_run_manifest.py` pattern: commit and dirty state of each component plus a hash
of the recipe, extended with dataset revisions and per-task content hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

from .config import BENCHFLOW_COMMIT, PipelineConfig
from .io import file_sha256, load_json

PIPELINE_COMMIT_ENV = "POSTTRAINARENA_PIPELINE_COMMIT"
PACKAGE_VERSIONS = (
    "posttrainarena-benchflow-pipeline",
    "benchflow",
    "trl",
    "peft",
    "transformers",
    "torch",
    "vllm",
    "accelerate",
    "datasets",
)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def package_source_sha256() -> str:
    """Digest of the imported pipeline package's Python sources (works without git)."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pipeline_source() -> dict[str, Any]:
    """Commit and dirty state of the checkout the pipeline package was imported from."""
    here = Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], here)
    status = _git(["status", "--porcelain", "--untracked-files=no"], here)
    return {
        "commit": commit or os.environ.get(PIPELINE_COMMIT_ENV) or None,
        "commit_source": (
            "git"
            if commit
            else ("environment" if os.environ.get(PIPELINE_COMMIT_ENV) else None)
        ),
        "dirty": None if status is None else bool(status),
        # Modified tracked files (for example the job script's vLLM all-reduce patch).
        "dirty_files": (
            None
            if status is None
            else [line[3:] for line in status.splitlines() if line.strip()]
        ),
        "package_source_sha256": package_source_sha256(),
    }


def installed_benchflow() -> dict[str, Any]:
    """The pinned BenchFlow commit and the one actually installed."""
    installed = None
    installed_version = None
    try:
        dist = distribution("benchflow")
        installed_version = dist.version
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        installed = (direct_url.get("vcs_info") or {}).get("commit_id")
    except (PackageNotFoundError, ValueError):
        pass
    return {
        "pinned_commit": BENCHFLOW_COMMIT,
        "installed_commit": installed,
        "installed_version": installed_version,
        "matches_pin": installed == BENCHFLOW_COMMIT if installed else None,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_VERSIONS:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def recipe_identity(
    config: PipelineConfig,
    *,
    train_task_ids: list[str],
    suite_task_ids: dict[str, list[str]],
) -> dict[str, Any]:
    """Everything that defines the recipe except machine-local paths, and its hash.

    Two runs with the same `recipe_sha256` used the same model, datasets, task lists and
    hyperparameters, wherever their config files and run directories lived.
    """
    recipe = {
        "model": config.model,
        "model_revision": config.model_revision,
        "train_dataset": {
            "repo_id": config.train_dataset.repo_id,
            "revision": config.train_dataset.revision,
            "path": config.train_dataset.path,
            "task_ids": list(train_task_ids),
        },
        "eval_suites": [
            {
                "name": suite.name,
                "repo_id": suite.dataset.repo_id,
                "revision": suite.dataset.revision,
                "path": suite.dataset.path,
                "task_ids": list(suite_task_ids[suite.name]),
            }
            for suite in config.suites
        ],
        "runtime": asdict(config.runtime),
        "harness": asdict(config.harness),
        "evaluation": asdict(config.evaluation),
        "teacher": asdict(config.teacher),
        "sft": asdict(config.sft),
        "grpo": asdict(config.grpo),
        "benchflow_commit": BENCHFLOW_COMMIT,
    }
    return {"recipe_sha256": _canonical_sha256(recipe), "recipe": recipe}


def _snapshot(reports: Path, label: str) -> dict[str, Any] | None:
    path = reports / f"{label}_snapshot_integrity.json"
    if not path.is_file():
        return None
    integrity = load_json(path)
    return {
        "resolved_revision": integrity.get("resolved_revision"),
        "marker_sha256": integrity.get("marker_sha256"),
        # Digest of each task package as trained or evaluated: relative paths and bytes of
        # every file, after the reference solution (oracle/, solution/) was removed.
        "task_sha256": integrity.get("task_sha256"),
        "reference_solutions_removed": len(
            integrity.get("reference_solutions_removed") or []
        ),
        "integrity_report": path.name,
    }


def _dataset_entry(
    *,
    role: str,
    name: str,
    label: str,
    dataset: Any,
    task_ids: list[str],
    reports: Path,
) -> dict[str, Any]:
    return {
        "role": role,
        "name": name,
        "repo_id": dataset.repo_id,
        "revision": dataset.revision,
        "path": dataset.path,
        "task_list_sha256": (
            file_sha256(dataset.task_list) if dataset.task_list.is_file() else None
        ),
        "task_count": len(task_ids),
        "snapshot": _snapshot(reports, label),
    }


def build_provenance(
    *,
    config: PipelineConfig,
    run_name: str,
    run_dir: Path,
    reports: Path,
    train_task_ids: list[str],
    suite_task_ids: dict[str, list[str]],
    suite_labels: dict[str, str],
    status: str,
    checkpoints: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    identity = recipe_identity(
        config, train_task_ids=train_task_ids, suite_task_ids=suite_task_ids
    )
    return {
        "schema_version": 1,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": status,
        "dry_run": dry_run,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "pipeline": pipeline_source(),
        "benchflow": installed_benchflow(),
        "packages": package_versions(),
        "python": platform.python_version(),
        "model": {"id": config.model, "revision": config.model_revision},
        "recipe": {
            "config_path": str(config.source),
            "config_file_sha256": (
                file_sha256(config.source) if config.source.is_file() else None
            ),
            "recipe_sha256": identity["recipe_sha256"],
        },
        "sampler": {
            "task_sampler": config.grpo.task_sampler,
            "seed": config.grpo.seed,
            "report": "train_sampler.json",
        },
        "datasets": [
            _dataset_entry(
                role="train",
                name="train",
                label="train",
                dataset=config.train_dataset,
                task_ids=train_task_ids,
                reports=reports,
            ),
            *(
                _dataset_entry(
                    role="eval",
                    name=suite.name,
                    label=suite_labels[suite.name],
                    dataset=suite.dataset,
                    task_ids=suite_task_ids[suite.name],
                    reports=reports,
                )
                for suite in config.suites
            ),
        ],
        "checkpoints": checkpoints or {},
    }
