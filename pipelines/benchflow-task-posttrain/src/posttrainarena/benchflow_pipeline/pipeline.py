"""End-to-end task-list orchestration for BenchFlow post-training."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BENCHFLOW_COMMIT, DatasetConfig, PipelineConfig
from .io import (
    CommandRunner,
    directory_sha256,
    file_sha256,
    load_json,
    load_score,
    read_task_ids,
    write_json,
)
from .layout import RunLayout


def utc_run_name() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _task_flags(task_ids: list[str]) -> list[str]:
    return [item for task_id in task_ids for item in ("--include-task", task_id)]


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _reference_sha256(reference: str, *, revision: str | None) -> str:
    path = Path(reference)
    if path.is_dir():
        return directory_sha256(path)
    return hashlib.sha256(f"{reference}@{revision or ''}".encode()).hexdigest()


def _teacher_sources_sha256(selection_path: Path) -> str:
    selection = load_json(selection_path)
    selected = selection.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"No selected teacher rollouts in {selection_path}")
    digest = hashlib.sha256()
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid selected teacher rollout in {selection_path}")
        task_id = row.get("task_id")
        rollout_value = row.get("rollout_dir")
        if not isinstance(task_id, str) or not isinstance(rollout_value, str):
            raise ValueError(f"Incomplete selected teacher rollout in {selection_path}")
        rollout_dir = Path(rollout_value)
        digest.update(task_id.encode())
        digest.update(b"\0")
        for relative in (
            Path("result.json"),
            Path("trajectory") / "llm_trajectory.jsonl",
        ):
            digest.update(str(relative).encode())
            digest.update(b"\0")
            digest.update(_sha256(rollout_dir / relative).encode())
    return digest.hexdigest()


class Pipeline:
    """Run or plan the complete SFT and conditional-GRPO workflow."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        run_name: str | None = None,
        dry_run: bool = False,
        resume: bool = False,
    ) -> None:
        self.config = config
        self.run_name = run_name or utc_run_name()
        self.layout = RunLayout(config.output_root / self.run_name)
        self.runner = CommandRunner(cwd=config.source.parent, dry_run=dry_run)
        self.dry_run = dry_run
        self.resume = resume
        self.train_task_ids = read_task_ids(config.train_dataset.task_list)
        self.eval_task_ids = read_task_ids(config.eval_dataset.task_list)
        if not self.train_task_ids or not self.eval_task_ids:
            raise ValueError("Training and eval task lists must both be non-empty")
        overlap = sorted(set(self.train_task_ids) & set(self.eval_task_ids))
        if overlap:
            raise ValueError(
                "Training and eval task IDs must be disjoint; overlap: "
                + ", ".join(overlap)
            )

    def plan(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_dir": str(self.layout.root),
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "benchflow_commit": BENCHFLOW_COMMIT,
            "train_dataset": asdict(self.config.train_dataset),
            "eval_dataset": asdict(self.config.eval_dataset),
            "train_task_count": len(self.train_task_ids),
            "eval_task_count": len(self.eval_task_ids),
            "train_task_ids": list(self.train_task_ids),
            "eval_task_ids": list(self.eval_task_ids),
            "runtime": asdict(self.config.runtime),
            "harness": asdict(self.config.harness),
            "evaluation": asdict(self.config.evaluation),
            "harness_migration": {
                "applied_stages": ["teacher", "evaluation", "grpo"],
                "pending_stages": [],
            },
            "teacher": asdict(self.config.teacher),
            "sft": asdict(self.config.sft),
            "grpo": asdict(self.config.grpo),
            "tracking": asdict(self.config.tracking),
            "stages": [
                "snapshot_train_tasks",
                "snapshot_eval_tasks",
                "validate_task_content_isolation",
                *(
                    ["sync_base_endpoint"]
                    if self.config.evaluation.sync_base_to_vllm
                    else []
                ),
                "baseline_eval",
                "collect_verified_teacher_rollouts",
                "convert_and_validate_sft_data",
                "train_sft",
                "sync_sft_endpoint",
                "sft_eval",
                "grpo_gate_eval",
                "conditional_grpo",
                "sync_grpo_endpoint",
                "final_eval",
                "compare_eval_lift",
                "write_score_report",
            ],
        }

    def run(self) -> dict[str, Any]:
        os.environ.setdefault("WANDB_PROJECT", self.config.tracking.project)
        self._prepare_run_plan()
        (self.layout.root / "train_task_ids.txt").write_text(
            "\n".join(self.train_task_ids) + "\n"
        )
        (self.layout.root / "eval_task_ids.txt").write_text(
            "\n".join(self.eval_task_ids) + "\n"
        )
        self._snapshot(
            "train",
            self.config.train_dataset,
            self.train_task_ids,
            self.layout.train_tasks,
        )
        self._snapshot(
            "eval", self.config.eval_dataset, self.eval_task_ids, self.layout.eval_tasks
        )
        if not self.dry_run:
            self._validate_task_content_isolation()
        baseline_path = self.layout.results / "baseline_eval.json"
        baseline_jobs = self.layout.jobs / "baseline"
        if self.config.evaluation.sync_base_to_vllm and not (
            self.resume and baseline_path.is_file()
        ):
            self._sync_base_endpoint()
        baseline_score = self._evaluate(
            stage="baseline_eval",
            model=self.config.model,
            tasks_dir=self.layout.eval_tasks,
            task_ids=self.eval_task_ids,
            jobs_dir=baseline_jobs,
            metrics_path=baseline_path,
            policy_sha256=_reference_sha256(
                self.config.model,
                revision=self.config.model_revision,
            ),
        )
        final_model = self.config.model
        final_score = baseline_score
        final_jobs = baseline_jobs
        sft_score: float | None = None
        grpo_gate_score: float | None = None
        grpo_planned = False
        grpo_ran = False
        if self.config.sft.enabled:
            self._collect_and_convert_teacher_data()
            final_model = str(self.layout.sft_merged)
            self._train_sft(final_model)
            self._sync_student_endpoint(
                checkpoint=Path(final_model),
                stage="sft",
            )
            sft_path = self.layout.results / "sft_eval.json"
            final_jobs = self.layout.jobs / "sft"
            sft_score = self._evaluate(
                stage="sft_eval",
                model=final_model,
                tasks_dir=self.layout.eval_tasks,
                task_ids=self.eval_task_ids,
                jobs_dir=final_jobs,
                metrics_path=sft_path,
                policy_sha256=self._policy_sha256(final_model),
            )
            final_score = sft_score
        if self.config.grpo.enabled:
            gate_ids = self.train_task_ids[: max(1, self.config.grpo.gate_task_count)]
            grpo_gate_score = self._evaluate(
                stage="grpo_gate_eval",
                model=final_model,
                tasks_dir=self.layout.train_tasks,
                task_ids=gate_ids,
                jobs_dir=self.layout.jobs / "grpo-gate",
                metrics_path=self.layout.results / "grpo_gate_eval.json",
                policy_sha256=self._policy_sha256(final_model),
            )
        if self._should_run_grpo(grpo_gate_score):
            grpo_planned = True
            grpo_ran = not self.dry_run
            grpo_input_model = final_model
            grpo_model = str(self.layout.grpo_merged)
            self._train_grpo(input_model=grpo_input_model, output_model=grpo_model)
            self._sync_student_endpoint(
                checkpoint=Path(grpo_model),
                stage="grpo",
            )
            final_model = grpo_model
            final_jobs = self.layout.jobs / "posttrain"
            final_score = self._evaluate(
                stage="posttrain_eval",
                model=final_model,
                tasks_dir=self.layout.eval_tasks,
                task_ids=self.eval_task_ids,
                jobs_dir=final_jobs,
                metrics_path=self.layout.results / "posttrain_eval.json",
                policy_sha256=self._policy_sha256(final_model),
            )
        if self.config.sft.enabled or grpo_planned:
            self.runner.run(
                "compare_eval_lift",
                [
                    "bench",
                    "eval",
                    "compare-lift",
                    "--baseline",
                    str(baseline_jobs),
                    "--trained",
                    str(final_jobs),
                    "--out",
                    str(self.layout.reports / "EVAL_LIFT.md"),
                    "--json-out",
                    str(self.layout.reports / "eval_lift.json"),
                ],
            )
        return self._write_score(
            baseline_score=baseline_score,
            sft_score=sft_score,
            grpo_gate_score=grpo_gate_score,
            final_score=final_score,
            final_model=final_model,
            grpo_planned=grpo_planned,
            grpo_ran=grpo_ran,
        )

    def _prepare_run_plan(self) -> None:
        plan_path = self.layout.reports / "plan.json"
        current = _json_normalized(self.plan())
        if self.resume and self.layout.root.exists():
            if not plan_path.is_file():
                raise RuntimeError(
                    "Cannot resume a run without reports/plan.json; use a new run "
                    "name or restore the original plan"
                )
            existing = load_json(plan_path)
            if existing != current:
                changed = [
                    key
                    for key in sorted(set(existing) | set(current))
                    if existing.get(key) != current.get(key)
                ]
                raise RuntimeError(
                    "Cannot resume an incompatible run plan; use a new run name or "
                    "restore the original recipe. Changed fields: " + ", ".join(changed)
                )
        self.layout.root.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, current)

    def _should_run_grpo(self, gate_score: float | None) -> bool:
        if not self.config.grpo.enabled:
            return False
        if self.dry_run:
            return True
        if self.config.grpo.run_policy == "always":
            return True
        return gate_score is not None and gate_score >= self.config.grpo.threshold

    def _snapshot(
        self,
        label: str,
        dataset: DatasetConfig,
        task_ids: list[str],
        destination: Path,
    ) -> None:
        marker = destination / ".benchflow-source.json"
        if self.resume and marker.is_file():
            self._validate_snapshot_integrity(
                label=label,
                dataset=dataset,
                task_ids=task_ids,
                destination=destination,
                marker=marker,
            )
            return
        self.runner.run(
            f"snapshot_{label}_tasks",
            [
                "bench",
                "tasks",
                "snapshot-hf",
                dataset.repo_id,
                str(destination),
                "--path",
                dataset.path,
                "--revision",
                dataset.revision,
                "--overwrite",
                *_task_flags(task_ids),
            ],
        )
        if not self.dry_run:
            self._write_snapshot_integrity(
                label=label,
                dataset=dataset,
                task_ids=task_ids,
                destination=destination,
                marker=marker,
            )

    def _snapshot_integrity_path(self, label: str) -> Path:
        return self.layout.reports / f"{label}_snapshot_integrity.json"

    def _snapshot_task_digests(
        self,
        *,
        task_ids: list[str],
        destination: Path,
    ) -> dict[str, str]:
        return {
            task_id: self._task_package_sha256(destination / task_id)
            for task_id in task_ids
        }

    def _write_snapshot_integrity(
        self,
        *,
        label: str,
        dataset: DatasetConfig,
        task_ids: list[str],
        destination: Path,
        marker: Path,
    ) -> None:
        marker_payload = load_json(marker)
        write_json(
            self._snapshot_integrity_path(label),
            {
                "schema_version": 1,
                "label": label,
                "dataset": {
                    "repo_id": dataset.repo_id,
                    "revision": dataset.revision,
                    "path": dataset.path,
                },
                "task_ids": task_ids,
                "marker_sha256": _sha256(marker),
                "resolved_revision": marker_payload.get("resolved_revision"),
                "task_sha256": self._snapshot_task_digests(
                    task_ids=task_ids,
                    destination=destination,
                ),
            },
        )

    def _validate_snapshot_integrity(
        self,
        *,
        label: str,
        dataset: DatasetConfig,
        task_ids: list[str],
        destination: Path,
        marker: Path,
    ) -> None:
        marker_payload = load_json(marker)
        expected_marker = {
            "repo": dataset.repo_id,
            "repo_type": "dataset",
            "path": dataset.path,
            "requested_revision": dataset.revision,
            "resolved_revision": dataset.revision,
            "include_tasks": task_ids,
            "dirty": False,
            "local_path": str(destination),
        }
        problems = [
            f"marker.{key}={marker_payload.get(key)!r}, expected {value!r}"
            for key, value in expected_marker.items()
            if marker_payload.get(key) != value
        ]
        integrity_path = self._snapshot_integrity_path(label)
        if not integrity_path.is_file():
            problems.append(f"missing snapshot integrity report: {integrity_path}")
        else:
            integrity = load_json(integrity_path)
            current_digests = self._snapshot_task_digests(
                task_ids=task_ids,
                destination=destination,
            )
            expected_integrity = {
                "schema_version": 1,
                "label": label,
                "dataset": {
                    "repo_id": dataset.repo_id,
                    "revision": dataset.revision,
                    "path": dataset.path,
                },
                "task_ids": task_ids,
                "marker_sha256": _sha256(marker),
                "resolved_revision": dataset.revision,
                "task_sha256": current_digests,
            }
            problems.extend(
                f"integrity.{key} does not match the current snapshot"
                for key, value in expected_integrity.items()
                if integrity.get(key) != value
            )
        if problems:
            raise RuntimeError(
                "Cannot resume incompatible task snapshot:\n- " + "\n- ".join(problems)
            )

    @staticmethod
    def _task_package_sha256(
        task_dir: Path,
        *,
        normalize_task_name: bool = False,
    ) -> str:
        if not task_dir.is_dir():
            raise FileNotFoundError(task_dir)
        files = sorted(path for path in task_dir.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"Empty task package: {task_dir}")
        digest = hashlib.sha256()
        for path in files:
            relative = str(path.relative_to(task_dir))
            content = path.read_bytes()
            if relative == "task.md" and normalize_task_name:
                text = content.decode("utf-8")
                text = re.sub(
                    r"(?m)^  name: [^\n]*\n",
                    "",
                    text,
                    count=1,
                )
                content = text.replace("\r\n", "\n").encode()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(content)
        return digest.hexdigest()

    def _validate_task_content_isolation(self) -> None:
        train_digests = {
            task_id: self._task_package_sha256(
                self.layout.train_tasks / task_id,
                normalize_task_name=True,
            )
            for task_id in self.train_task_ids
        }
        eval_digests = {
            task_id: self._task_package_sha256(
                self.layout.eval_tasks / task_id,
                normalize_task_name=True,
            )
            for task_id in self.eval_task_ids
        }
        by_train_digest = {digest: task_id for task_id, digest in train_digests.items()}
        overlaps = [
            (by_train_digest[digest], eval_task_id)
            for eval_task_id, digest in eval_digests.items()
            if digest in by_train_digest
        ]
        if overlaps:
            formatted = ", ".join(
                f"{train_id} == {eval_id}" for train_id, eval_id in overlaps
            )
            raise RuntimeError(
                "Training and evaluation task packages overlap by canonical "
                f"content digest: {formatted}"
            )
        write_json(
            self.layout.reports / "task_content_isolation.json",
            {
                "schema_version": 1,
                "algorithm": "sha256-relative-files-task-name-normalized",
                "train_task_count": len(train_digests),
                "eval_task_count": len(eval_digests),
                "overlap_count": 0,
            },
        )

    def _evaluate(
        self,
        *,
        stage: str,
        model: str,
        tasks_dir: Path,
        task_ids: list[str],
        jobs_dir: Path,
        metrics_path: Path,
        policy_sha256: str,
    ) -> float | None:
        if self.resume and metrics_path.is_file():
            return self._load_resumed_evaluation(
                model=model,
                task_ids=task_ids,
                jobs_dir=jobs_dir,
                metrics_path=metrics_path,
                policy_sha256=policy_sha256,
            )
        if self.resume:
            if jobs_dir.exists():
                shutil.rmtree(jobs_dir)
            for artifact in (
                metrics_path.with_name(f"{metrics_path.stem}_health.json"),
                metrics_path.with_name(f"{metrics_path.stem}_task_manifest.json"),
                metrics_path.with_name(f"{metrics_path.stem}_run_config.json"),
            ):
                artifact.unlink(missing_ok=True)
        from .opencode import evaluate

        payload = evaluate(
            config=self.config,
            runner=self.runner,
            stage=stage,
            model=model,
            tasks_dir=tasks_dir,
            task_ids=task_ids,
            jobs_dir=jobs_dir,
            metrics_path=metrics_path,
            policy_sha256=policy_sha256,
        )
        score = payload["score"]
        return None if score is None else float(score)

    def _load_resumed_evaluation(
        self,
        *,
        model: str,
        task_ids: list[str],
        jobs_dir: Path,
        metrics_path: Path,
        policy_sha256: str,
    ) -> float:
        from .opencode import load_summary, served_model

        payload = load_json(metrics_path)
        expected = {
            "mode": "eval",
            "harness": self.config.harness.agent,
            "model": model,
            "served_model": served_model(self.config, model),
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "jobs_dir": str(jobs_dir),
            "capture_token_logprobs": False,
            "policy_sha256": policy_sha256,
        }
        problems = [
            f"{key}={payload.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        loaded = load_summary(
            jobs_dir=jobs_dir,
            health_path=metrics_path.with_name(f"{metrics_path.stem}_health.json"),
            expected_tasks=len(task_ids),
            expected_task_ids=task_ids,
        )
        score = load_score(metrics_path)
        if not math.isclose(score, float(loaded["score"]), rel_tol=0, abs_tol=1e-12):
            problems.append(
                f"saved score={score!r}, artifact score={loaded['score']!r}"
            )
        if problems:
            raise RuntimeError(
                "Cannot resume incompatible evaluation artifacts:\n- "
                + "\n- ".join(problems)
            )
        return score

    def _policy_sha256(self, reference: str) -> str:
        path = Path(reference)
        if path.is_dir():
            return directory_sha256(path)
        if reference == self.config.model:
            return _reference_sha256(
                reference,
                revision=self.config.model_revision,
            )
        if self.dry_run:
            return f"pending:{reference}"
        raise FileNotFoundError(reference)

    def _collect_and_convert_teacher_data(self) -> None:
        manifest_path = self.layout.reports / "teacher_manifest.json"
        reuse_teacher = (
            self.resume
            and manifest_path.is_file()
            and self.layout.teacher_selection.is_file()
        )
        if reuse_teacher:
            self._validate_resumed_teacher_state(manifest_path)
        else:
            from .teacher import collect_verified_teacher_rollouts

            collect_verified_teacher_rollouts(
                config=self.config,
                runner=self.runner,
                tasks_dir=self.layout.train_tasks,
                task_ids=self.train_task_ids,
                jobs_dir=self.layout.jobs / "teacher",
                manifest_path=manifest_path,
                selection_path=self.layout.teacher_selection,
            )
        conversion_manifest = self.layout.reports / "sft_conversion.json"
        selection_digest = (
            None if self.dry_run else _sha256(self.layout.teacher_selection)
        )
        source_digest = (
            None
            if self.dry_run
            else _teacher_sources_sha256(self.layout.teacher_selection)
        )
        reuse_conversion = (
            self.resume
            and self.layout.sft_jsonl.is_file()
            and conversion_manifest.is_file()
            and selection_digest is not None
            and source_digest is not None
            and self._conversion_matches_selection(
                conversion_manifest,
                selection_digest=selection_digest,
                source_digest=source_digest,
            )
        )
        if not reuse_conversion:
            self.runner.run(
                "convert_verified_sft_data",
                [
                    "bench",
                    "train",
                    "convert",
                    str(self.layout.jobs / "teacher"),
                    "--format",
                    "trl-sft",
                    "--out",
                    str(self.layout.sft_jsonl),
                    "--min-reward",
                    str(self.config.teacher.min_reward),
                    "--row-mode",
                    "exchange",
                    "--canonical-selection",
                    str(self.layout.teacher_selection),
                    "--context-policy",
                    "message-window",
                    "--tokenizer",
                    self.config.model,
                    *(
                        [
                            "--tokenizer-revision",
                            self.config.model_revision,
                        ]
                        if self.config.model_revision
                        else []
                    ),
                    "--max-length",
                    str(self.config.sft.max_length),
                    "--manifest",
                    str(conversion_manifest),
                ],
            )
            if not self.dry_run:
                conversion = load_json(conversion_manifest)
                conversion["teacher_selection_sha256"] = selection_digest
                conversion["teacher_sources_sha256"] = source_digest
                conversion["sft_jsonl_sha256"] = _sha256(self.layout.sft_jsonl)
                write_json(conversion_manifest, conversion)
        self.runner.run(
            "validate_verified_sft_data",
            [
                "bench",
                "train",
                "validate",
                str(self.layout.sft_jsonl),
                "--format",
                "trl-sft",
                "--source-jobs",
                str(self.layout.jobs / "teacher"),
                "--source-canonical-selection",
                str(self.layout.teacher_selection),
                "--require-llm-trajectory",
                "--require-tool-calls",
                "--tokenizer",
                self.config.model,
                *(
                    [
                        "--tokenizer-revision",
                        self.config.model_revision,
                    ]
                    if self.config.model_revision
                    else []
                ),
                "--max-length",
                str(self.config.sft.max_length),
            ],
        )

    def _conversion_matches_selection(
        self,
        conversion_manifest: Path,
        *,
        selection_digest: str,
        source_digest: str,
    ) -> bool:
        try:
            conversion = load_json(conversion_manifest)
            return (
                conversion.get("teacher_selection_sha256") == selection_digest
                and conversion.get("teacher_sources_sha256") == source_digest
                and conversion.get("sft_jsonl_sha256") == _sha256(self.layout.sft_jsonl)
            )
        except (OSError, ValueError):
            return False

    def _validate_resumed_teacher_state(self, manifest_path: Path) -> None:
        manifest = load_json(manifest_path)
        selection = load_json(self.layout.teacher_selection)
        required = (
            len(self.train_task_ids)
            if self.config.teacher.require_all_tasks
            else self.config.teacher.min_verified
        )
        expected = {
            "teacher_model": self.config.teacher.model,
            "teacher_source_model": self.config.teacher.source_model,
            "teacher_source_revision": self.config.teacher.source_revision,
            "requested_task_count": len(self.train_task_ids),
            "requested_task_ids": self.train_task_ids,
            "required_verified_count": required,
            "require_all_tasks": self.config.teacher.require_all_tasks,
        }
        problems = [
            f"{key}={manifest.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if manifest.get(key) != value
        ]
        verified_count = manifest.get("verified_count")
        selected_count = selection.get("selected_count")
        verified_rows = manifest.get("verified")
        selected_rows = selection.get("selected")
        if not isinstance(verified_rows, list) or any(
            not isinstance(row, dict) for row in verified_rows
        ):
            problems.append("manifest.verified must be a list of objects")
            verified_rows = []
        if not isinstance(selected_rows, list) or any(
            not isinstance(row, dict) for row in selected_rows
        ):
            problems.append("selection.selected must be a list of objects")
            selected_rows = []
        if not isinstance(verified_count, int) or verified_count < required:
            problems.append(
                f"verified_count={verified_count!r}, required at least {required}"
            )
        if verified_count != len(verified_rows):
            problems.append(
                f"manifest verified_count={verified_count!r}, "
                f"verified rows={len(verified_rows)}"
            )
        if selected_count != len(selected_rows):
            problems.append(
                f"selection selected_count={selected_count!r}, "
                f"selected rows={len(selected_rows)}"
            )
        if selected_count != verified_count:
            problems.append(
                f"selection.selected_count={selected_count!r}, "
                f"manifest verified_count={verified_count!r}"
            )
        if _json_normalized(selected_rows) != _json_normalized(verified_rows):
            problems.append("selection.selected does not match manifest.verified")
        selected_task_ids = [
            row.get("task_id") for row in selected_rows if isinstance(row, dict)
        ]
        if any(not isinstance(task_id, str) for task_id in selected_task_ids):
            problems.append("selection contains an invalid task_id")
        elif len(selected_task_ids) != len(set(selected_task_ids)):
            problems.append("selection contains duplicate task IDs")
        else:
            unexpected = sorted(set(selected_task_ids) - set(self.train_task_ids))
            if unexpected:
                problems.append(
                    "selection contains tasks outside the run plan: "
                    + ", ".join(unexpected)
                )
            if self.config.teacher.require_all_tasks and set(selected_task_ids) != set(
                self.train_task_ids
            ):
                problems.append(
                    "selection does not cover every requested training task"
                )
        teacher_root = (self.layout.jobs / "teacher").resolve()
        for row in selected_rows:
            rollout_value = row.get("rollout_dir")
            if not isinstance(rollout_value, str) or not rollout_value:
                problems.append("selection contains a missing rollout_dir")
                continue
            rollout_dir = Path(rollout_value).resolve()
            if not rollout_dir.is_relative_to(teacher_root):
                problems.append(
                    f"selected rollout is outside teacher jobs: {rollout_dir}"
                )
                continue
            for artifact in (
                rollout_dir / "result.json",
                rollout_dir / "trajectory" / "llm_trajectory.jsonl",
            ):
                if not artifact.is_file():
                    problems.append(f"selected rollout artifact is missing: {artifact}")
            reward = row.get("reward")
            if (
                not isinstance(reward, int | float)
                or isinstance(reward, bool)
                or float(reward) < self.config.teacher.min_reward
            ):
                problems.append(
                    "selected rollout has invalid reward for "
                    f"{row.get('task_id')!r}: {reward!r}"
                )
            if row.get("training_ready") is not True:
                problems.append(
                    f"selected rollout is not training-ready: {row.get('task_id')!r}"
                )
        if problems:
            raise RuntimeError(
                "Cannot resume incompatible teacher state; use a new run name or "
                "restore the original recipe:\n- " + "\n- ".join(problems)
            )

    def _train_sft(self, output_model: str) -> None:
        metrics = Path(output_model) / "train_metrics.json"
        if (
            self.resume
            and metrics.is_file()
            and self._sft_checkpoint_is_current(
                metrics,
                output_model=Path(output_model),
            )
        ):
            return
        if self.resume:
            for path in (self.layout.sft_adapter, Path(output_model)):
                if path.exists():
                    shutil.rmtree(path)
        if self.dry_run:
            self.runner.commands.append(
                {
                    "name": "train_sft",
                    "call": "sft.train_sft",
                    "train_jsonl": str(self.layout.sft_jsonl),
                    "output_dir": output_model,
                }
            )
            return
        from .sft import train_sft

        train_sft(
            config=self.config,
            train_jsonl=self.layout.sft_jsonl,
            adapter_dir=self.layout.sft_adapter,
            output_dir=Path(output_model),
            run_name=self.run_name,
        )

    def _sft_checkpoint_is_current(
        self,
        metrics_path: Path,
        *,
        output_model: Path,
    ) -> bool:
        try:
            metrics = load_json(metrics_path)
            return (
                metrics.get("mode") == "sft"
                and metrics.get("base_model") == self.config.model
                and metrics.get("model_revision") == self.config.model_revision
                and metrics.get("adapter_dir") == str(self.layout.sft_adapter)
                and metrics.get("merged_model_dir") == str(output_model)
                and metrics.get("train_jsonl_sha256") == _sha256(self.layout.sft_jsonl)
                and metrics.get("adapter_sha256")
                == directory_sha256(self.layout.sft_adapter)
                and metrics.get("merged_model_sha256") == directory_sha256(output_model)
            )
        except (OSError, ValueError):
            return False

    def _train_grpo(self, *, input_model: str, output_model: str) -> None:
        metrics = Path(output_model) / "train_metrics.json"
        if (
            self.resume
            and metrics.is_file()
            and self._grpo_checkpoint_is_current(
                metrics,
                input_model=input_model,
                output_model=Path(output_model),
            )
        ):
            return
        jobs_dir = self.layout.jobs / "grpo-train"
        if self.resume:
            for path in (jobs_dir, self.layout.grpo_adapter, Path(output_model)):
                if path.exists():
                    shutil.rmtree(path)
        if self.dry_run:
            self.runner.commands.append(
                {
                    "name": "train_grpo",
                    "call": "grpo.train_grpo",
                    "model": input_model,
                    "output_dir": output_model,
                    "resume_policy": "restart-stage",
                }
            )
            return
        from .grpo import train_grpo

        train_grpo(
            config=self.config,
            model=input_model,
            tasks_dir=self.layout.train_tasks,
            task_ids=self.train_task_ids,
            jobs_dir=jobs_dir,
            adapter_dir=self.layout.grpo_adapter,
            output_dir=Path(output_model),
            run_name=f"{self.run_name}-grpo",
        )

    def _grpo_checkpoint_is_current(
        self,
        metrics_path: Path,
        *,
        input_model: str,
        output_model: Path,
    ) -> bool:
        revision = (
            self.config.model_revision if input_model == self.config.model else None
        )
        try:
            metrics = load_json(metrics_path)
            return (
                metrics.get("mode") == "grpo"
                and metrics.get("model") == input_model
                and metrics.get("task_ids") == self.train_task_ids
                and metrics.get("adapter_dir") == str(self.layout.grpo_adapter)
                and metrics.get("merged_model_dir") == str(output_model)
                and metrics.get("base_checkpoint_sha256")
                == _reference_sha256(input_model, revision=revision)
                and metrics.get("adapter_sha256")
                == directory_sha256(self.layout.grpo_adapter)
                and metrics.get("merged_model_sha256") == directory_sha256(output_model)
            )
        except (OSError, ValueError):
            return False

    def _sync_student_endpoint(self, *, checkpoint: Path, stage: str) -> None:
        report_path = self.layout.results / f"{stage}_endpoint_sync.json"
        if self.dry_run:
            self.runner.commands.append(
                {
                    "name": f"sync_{stage}_endpoint",
                    "call": "grpo.sync_checkpoint_to_vllm",
                    "checkpoint": str(checkpoint),
                    "vllm_server_base_url_env": (
                        self.config.grpo.vllm_server_base_url_env
                    ),
                    "policy_attestation": True,
                }
            )
            return
        from .grpo import attest_served_policy, sync_checkpoint_to_vllm

        payload = sync_checkpoint_to_vllm(
            config=self.config,
            checkpoint=checkpoint,
        )
        write_json(
            report_path,
            {
                **payload,
                "policy_attestation": attest_served_policy(
                    config=self.config,
                    model_role="student",
                ),
            },
        )

    def _sync_base_endpoint(self) -> None:
        report_path = self.layout.results / "base_endpoint_sync.json"
        if self.dry_run:
            self.runner.commands.append(
                {
                    "name": "sync_base_endpoint",
                    "call": "grpo.sync_reference_to_vllm",
                    "reference": self.config.model,
                    "revision": self.config.model_revision,
                    "vllm_server_base_url_env": (
                        self.config.grpo.vllm_server_base_url_env
                    ),
                    "policy_attestation": True,
                }
            )
            return
        from .grpo import attest_served_policy, sync_reference_to_vllm

        payload = sync_reference_to_vllm(
            config=self.config,
            reference=self.config.model,
        )
        write_json(
            report_path,
            {
                **payload,
                "model_revision": self.config.model_revision,
                "policy_attestation": attest_served_policy(
                    config=self.config,
                    model_role="base",
                ),
            },
        )

    def _write_score(
        self,
        *,
        baseline_score: float | None,
        sft_score: float | None,
        grpo_gate_score: float | None,
        final_score: float | None,
        final_model: str,
        grpo_planned: bool,
        grpo_ran: bool,
    ) -> dict[str, Any]:
        delta = (
            None
            if baseline_score is None or final_score is None
            else final_score - baseline_score
        )
        summary = {
            "schema_version": 1,
            "run_name": self.run_name,
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "final_model": final_model,
            "train_task_ids": self.train_task_ids,
            "eval_task_ids": self.eval_task_ids,
            "baseline_score": baseline_score,
            "sft_score": sft_score,
            "grpo_gate_score": grpo_gate_score,
            "score_after_posttrain": final_score,
            "delta_score": delta,
            "grpo_threshold": self.config.grpo.threshold,
            "grpo_run_policy": self.config.grpo.run_policy,
            "grpo_planned": grpo_planned,
            "grpo_ran": grpo_ran,
            "checkpoints": {
                "sft_adapter": (
                    str(self.layout.sft_adapter) if self.config.sft.enabled else None
                ),
                "sft_merged": (
                    str(self.layout.sft_merged) if self.config.sft.enabled else None
                ),
                "grpo_adapter": (
                    str(self.layout.grpo_adapter) if grpo_planned else None
                ),
                "grpo_merged": (str(self.layout.grpo_merged) if grpo_planned else None),
            },
            "harness": asdict(self.config.harness),
            "evaluation": asdict(self.config.evaluation),
            "teacher": asdict(self.config.teacher),
            "sft": asdict(self.config.sft),
            "grpo": asdict(self.config.grpo),
            "tracking": asdict(self.config.tracking),
            "harness_migration": {
                "applied_stages": ["teacher", "evaluation", "grpo"],
                "pending_stages": [],
            },
            "benchflow_commit": BENCHFLOW_COMMIT,
            "train_dataset": asdict(self.config.train_dataset),
            "eval_dataset": asdict(self.config.eval_dataset),
            "commands": self.runner.commands,
        }
        write_json(self.layout.reports / "score.json", summary)
        self.layout.reports.mkdir(parents=True, exist_ok=True)
        (self.layout.reports / "SCORE.md").write_text(
            "\n".join(
                [
                    "# Post-Training Score",
                    "",
                    f"- Base model: `{self.config.model}`",
                    f"- Final model: `{final_model}`",
                    f"- Baseline score: `{baseline_score}`",
                    f"- SFT score: `{sft_score}`",
                    f"- GRPO gate score: `{grpo_gate_score}`",
                    f"- Final score: `{final_score}`",
                    f"- Delta: `{delta}`",
                    f"- GRPO planned: `{grpo_planned}`",
                    f"- GRPO ran: `{grpo_ran}`",
                    f"- Training tasks: `{len(self.train_task_ids)}`",
                    f"- Eval tasks: `{len(self.eval_task_ids)}`",
                    "",
                ]
            )
        )
        return summary

    def reset_run(self) -> None:
        if self.layout.root.exists():
            shutil.rmtree(self.layout.root)
