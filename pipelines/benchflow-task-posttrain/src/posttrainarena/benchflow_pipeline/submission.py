"""Convert a validated team entry into a pinned training dataset and recipe."""

from __future__ import annotations

import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_json


REQUIRED_FIELDS = ("team_name", "contact_email", "track")
SAFE_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class PreparedSubmission:
    submission_id: str
    team_name: str
    dataset_repo: str
    dataset_revision: str
    task_count: int
    recipe_path: Path
    manifest_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "team_name": self.team_name,
            "dataset_repo": self.dataset_repo,
            "dataset_revision": self.dataset_revision,
            "task_count": self.task_count,
            "recipe_path": str(self.recipe_path),
            "manifest_path": str(self.manifest_path),
        }


def parse_flat_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith((" ", "\t", "-")) or ":" not in line:
            raise ValueError(
                f"{path}:{line_number}: submission.yaml must use flat key: value lines"
            )
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        values[key] = raw_value.split("#", 1)[0].strip().strip("\"'")
    return values


def submission_slug(value: str) -> str:
    slug = SAFE_SLUG.sub("-", value.lower()).strip("-")
    if not slug:
        raise ValueError("submission name does not contain a usable slug")
    return slug


def _load_entry(entry_dir: Path) -> tuple[dict[str, str], list[Path]]:
    manifest_path = entry_dir / "submission.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = parse_flat_yaml(manifest_path)
    missing = [field for field in REQUIRED_FIELDS if not manifest.get(field)]
    if missing:
        raise ValueError(f"submission.yaml missing fields: {', '.join(missing)}")
    if manifest["track"] != "environments":
        raise ValueError("managed SFT/GRPO currently supports track: environments")
    tasks_dir = entry_dir / "envs"
    tasks = (
        sorted(path for path in tasks_dir.iterdir() if (path / "task.md").is_file())
        if tasks_dir.is_dir()
        else []
    )
    if not tasks:
        raise ValueError(f"No task.md packages found under {tasks_dir}")
    if len(tasks) > 200:
        raise ValueError("environment submission exceeds the 200-task maximum")
    issues: dict[str, list[str]] = {}
    for task in tasks:
        task_issues = _task_issues(task)
        if task_issues:
            issues[task.name] = task_issues
    if issues:
        rendered = "; ".join(
            f"{task}: {', '.join(task_issues)}" for task, task_issues in issues.items()
        )
        raise ValueError(f"invalid task packages: {rendered}")
    return manifest, tasks


def _task_issues(task_dir: Path) -> list[str]:
    issues: list[str] = []
    required = (
        "task.md",
        "environment/Dockerfile",
        "verifier/test.sh",
        "oracle/solve.sh",
    )
    for relative in required:
        if not (task_dir / relative).is_file():
            issues.append(f"missing {relative}")
    task_md = task_dir / "task.md"
    if task_md.is_file():
        text = task_md.read_text(errors="replace")
        if not text.startswith("---\n"):
            issues.append("task.md missing YAML frontmatter")
        if "## prompt" not in text.lower():
            issues.append("task.md missing ## prompt")
    return issues


def _portable_recipe(
    *,
    base_config_path: Path,
    output_path: Path,
    dataset_repo: str,
    dataset_revision: str,
    train_task_ids: list[str],
) -> None:
    import tomli_w

    data = tomllib.loads(base_config_path.read_text())
    task_lists = output_path.parent / "task-lists"
    task_lists.mkdir(parents=True, exist_ok=True)
    train_list = task_lists / "train.txt"
    train_list.write_text("\n".join(train_task_ids) + "\n")
    eval_source = Path(data["eval_dataset"]["task_list"]).expanduser()
    if not eval_source.is_absolute():
        eval_source = (base_config_path.parent / eval_source).resolve()
    eval_list = task_lists / "eval.txt"
    shutil.copy2(eval_source, eval_list)
    data["train_dataset"].update(
        {
            "repo_id": dataset_repo,
            "revision": dataset_revision,
            "path": "tasks",
            "task_list": "task-lists/train.txt",
        }
    )
    data["eval_dataset"]["task_list"] = "task-lists/eval.txt"
    teacher = data.setdefault("teacher", {})
    teacher["min_verified"] = len(train_task_ids)
    teacher["require_all_tasks"] = True
    data.setdefault("output", {})["root"] = "runs"
    output_path.write_text(tomli_w.dumps(data))


def prepare_submission(
    *,
    entry_dir: Path,
    base_config_path: Path,
    output_dir: Path,
    dataset_repo: str,
    dataset_revision: str | None = None,
    upload: bool = False,
    private: bool = True,
    token: str | None = None,
) -> PreparedSubmission:
    manifest, tasks = _load_entry(entry_dir)
    submission_id = submission_slug(entry_dir.name)
    staging = output_dir / "dataset"
    shutil.rmtree(staging, ignore_errors=True)
    tasks_out = staging / "tasks"
    tasks_out.mkdir(parents=True)
    for task in tasks:
        shutil.copytree(task, tasks_out / task.name)
    dataset_manifest = {
        "schema_version": 1,
        "submission_id": submission_id,
        "team_name": manifest["team_name"],
        "track": manifest["track"],
        "task_ids": [task.name for task in tasks],
        "source_entry": entry_dir.name,
    }
    write_json(staging / "submission.json", dataset_manifest)
    if upload:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            dataset_repo,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        api.update_repo_settings(
            dataset_repo,
            repo_type="dataset",
            private=private,
        )
        commit = api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=staging,
            commit_message=f"Publish PostTrain Arena submission {submission_id}",
        )
        dataset_revision = commit.oid
    if not dataset_revision:
        raise ValueError("dataset_revision is required unless --upload is used")
    recipe_path = output_dir / "recipe.toml"
    _portable_recipe(
        base_config_path=base_config_path,
        output_path=recipe_path,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        train_task_ids=[task.name for task in tasks],
    )
    prepared = PreparedSubmission(
        submission_id=submission_id,
        team_name=manifest["team_name"],
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        task_count=len(tasks),
        recipe_path=recipe_path,
        manifest_path=output_dir / "prepared-submission.json",
    )
    write_json(prepared.manifest_path, prepared.as_dict())
    return prepared
