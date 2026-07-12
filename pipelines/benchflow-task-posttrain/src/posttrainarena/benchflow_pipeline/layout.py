"""Stable run-directory contract shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def train_tasks(self) -> Path:
        return self.root / "data" / "train"

    @property
    def eval_tasks(self) -> Path:
        return self.root / "data" / "eval"

    @property
    def sft_jsonl(self) -> Path:
        return self.root / "data" / "verified_teacher_sft.jsonl"

    @property
    def teacher_selection(self) -> Path:
        return self.root / "reports" / "teacher_selection.json"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"
