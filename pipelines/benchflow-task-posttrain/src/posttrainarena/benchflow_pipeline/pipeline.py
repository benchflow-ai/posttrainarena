"""End-to-end task-list orchestration for BenchFlow post-training."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BENCHFLOW_COMMIT, DatasetConfig, PipelineConfig
from .io import CommandRunner, load_score, read_task_ids, write_json
from .layout import RunLayout


def utc_run_name() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _task_flags(task_ids: list[str]) -> list[str]:
    return [item for task_id in task_ids for item in ("--include-task", task_id)]


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
        self.layout.root.mkdir(parents=True, exist_ok=True)
        write_json(self.layout.reports / "plan.json", self.plan())
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
        baseline_path = self.layout.results / "baseline_eval.json"
        baseline_jobs = self.layout.jobs / "baseline"
        baseline_score = self._evaluate(
            stage="baseline_eval",
            model=self.config.model,
            tasks_dir=self.layout.eval_tasks,
            task_ids=self.eval_task_ids,
            jobs_dir=baseline_jobs,
            metrics_path=baseline_path,
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
            final_model = str(self.layout.checkpoints / "sft-merged")
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
            )
        if self._should_run_grpo(grpo_gate_score):
            grpo_planned = True
            grpo_ran = not self.dry_run
            grpo_input_model = final_model
            grpo_model = str(self.layout.checkpoints / "grpo")
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

    def _evaluate(
        self,
        *,
        stage: str,
        model: str,
        tasks_dir: Path,
        task_ids: list[str],
        jobs_dir: Path,
        metrics_path: Path,
    ) -> float | None:
        if self.resume and metrics_path.is_file():
            return load_score(metrics_path)
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
        )
        score = payload["score"]
        return None if score is None else float(score)

    def _collect_and_convert_teacher_data(self) -> None:
        manifest_path = self.layout.reports / "teacher_manifest.json"
        if not (
            self.resume
            and manifest_path.is_file()
            and self.layout.teacher_selection.is_file()
        ):
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
        if not (
            self.resume
            and self.layout.sft_jsonl.is_file()
            and conversion_manifest.is_file()
        ):
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

    def _train_sft(self, output_model: str) -> None:
        metrics = Path(output_model) / "train_metrics.json"
        if self.resume and metrics.is_file():
            return
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
            adapter_dir=self.layout.checkpoints / "sft-adapter",
            output_dir=Path(output_model),
            run_name=self.run_name,
        )

    def _train_grpo(self, *, input_model: str, output_model: str) -> None:
        metrics = Path(output_model) / "train_metrics.json"
        if self.resume and metrics.is_file():
            return
        if self.dry_run:
            self.runner.commands.append(
                {
                    "name": "train_grpo",
                    "call": "grpo.train_grpo",
                    "model": input_model,
                    "output_dir": output_model,
                }
            )
            return
        from .grpo import train_grpo

        train_grpo(
            config=self.config,
            model=input_model,
            tasks_dir=self.layout.train_tasks,
            task_ids=self.train_task_ids,
            jobs_dir=self.layout.jobs / "grpo-train",
            output_dir=Path(output_model),
            run_name=f"{self.run_name}-grpo",
        )

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
                }
            )
            return
        from .grpo import sync_checkpoint_to_vllm

        payload = sync_checkpoint_to_vllm(
            config=self.config,
            checkpoint=checkpoint,
        )
        write_json(report_path, payload)

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
            "harness": asdict(self.config.harness),
            "evaluation": asdict(self.config.evaluation),
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
