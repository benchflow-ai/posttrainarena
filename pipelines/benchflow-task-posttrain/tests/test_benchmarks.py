from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.benchmarks import (
    load_benchmark_manifest,
    run_benchmark_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_manifest_and_dry_run_matrix(tmp_path: Path) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("0000_369_369503_qa_1\n")
    manifest = tmp_path / "benchmarks.toml"
    manifest.write_text(
        "[[benchmarks]]\n"
        'name = "data-agent"\n'
        'repo_id = "benchflow/data_agent_rl_environment_eval"\n'
        'revision = "0ea976c79e3248c85737c4f7363484e4d47ce287"\n'
        'path = "tasks"\n'
        'task_list = "tasks.txt"\n'
        "weight = 2.0\n"
    )
    run_dir = tmp_path / "run"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "reports" / "score.json").write_text(
        json.dumps({"final_model": "/tmp/final"})
    )

    suites = load_benchmark_manifest(manifest)
    result = run_benchmark_matrix(
        config_path=ROOT / "configs/qwen3-4b-data-agent-forced-grpo-smoke.toml",
        run_dir=run_dir,
        manifest_path=manifest,
        dry_run=True,
    )

    assert suites[0].name == "data-agent"
    assert result["benchmark_count"] == 1
    assert result["macro_delta_score"] is None
    assert result["benchmarks"][0]["task_ids"] == ["0000_369_369503_qa_1"]
    assert len(result["commands"]) == 3
    for command in result["commands"][1:]:
        assert command["command"][command["command"].index("--agent") + 1] == (
            "opencode"
        )


def test_checked_in_multi_benchmark_manifest_is_cross_domain() -> None:
    suites = load_benchmark_manifest(ROOT / "configs/multi-benchmark-smoke.toml")

    assert [suite.name for suite in suites] == ["data-agent", "skillsbench"]
    assert suites[0].repo_id == "benchflow/data_agent_rl_environment_eval"
    assert suites[1].repo_id == "benchflow/skillsbench"
    assert suites[1].path == ""


def test_benchmark_dry_run_defaults_to_merged_grpo_checkpoint(
    tmp_path: Path,
) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("0000_369_369503_qa_1\n")
    manifest = tmp_path / "benchmarks.toml"
    manifest.write_text(
        "[[benchmarks]]\n"
        'name = "data-agent"\n'
        'repo_id = "benchflow/data_agent_rl_environment_eval"\n'
        'revision = "0ea976c79e3248c85737c4f7363484e4d47ce287"\n'
        'path = "tasks"\n'
        'task_list = "tasks.txt"\n'
    )
    run_dir = tmp_path / "run"

    result = run_benchmark_matrix(
        config_path=ROOT / "configs/qwen3-4b-data-agent-forced-grpo-smoke.toml",
        run_dir=run_dir,
        manifest_path=manifest,
        dry_run=True,
    )

    assert result["final_model"].endswith("/checkpoints/grpo-merged")


def test_benchmark_matrix_restores_base_then_final_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_list = tmp_path / "tasks.txt"
    task_list.write_text("0000_369_369503_qa_1\n")
    manifest = tmp_path / "benchmarks.toml"
    manifest.write_text(
        "[[benchmarks]]\n"
        'name = "data-agent"\n'
        'repo_id = "benchflow/data_agent_rl_environment_eval"\n'
        'revision = "0ea976c79e3248c85737c4f7363484e4d47ce287"\n'
        'path = "tasks"\n'
        'task_list = "tasks.txt"\n'
    )
    run_dir = tmp_path / "run"
    final_model = run_dir / "checkpoints" / "grpo-merged"
    final_model.mkdir(parents=True)
    (run_dir / "reports").mkdir()
    (run_dir / "reports" / "score.json").write_text(
        json.dumps({"final_model": str(final_model)})
    )
    sequence: list[str] = []

    def fake_run(
        self,
        name: str,
        command: list[str],
        *,
        check: bool = True,
        env_overrides: dict[str, str] | None = None,
    ) -> int:
        del check, env_overrides
        self.commands.append({"name": name, "command": command})
        return 0

    def fake_evaluate(**kwargs: Any) -> dict[str, float]:
        sequence.append(f"eval:{kwargs['stage']}")
        return {"score": 0.0 if "baseline" in kwargs["stage"] else 1.0}

    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.benchmarks.CommandRunner.run",
        fake_run,
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.opencode.evaluate",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.grpo.sync_reference_to_vllm",
        lambda **_kwargs: (sequence.append("sync:base") or {"reference": "Qwen/base"}),
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.grpo.sync_checkpoint_to_vllm",
        lambda **_kwargs: (
            sequence.append("sync:final") or {"checkpoint": str(final_model)}
        ),
    )

    result = run_benchmark_matrix(
        config_path=ROOT / "configs/qwen3-4b-data-agent-forced-grpo-smoke.toml",
        run_dir=run_dir,
        manifest_path=manifest,
    )

    assert sequence == [
        "sync:base",
        "eval:baseline_benchmark_data-agent",
        "sync:final",
        "eval:final_benchmark_data-agent",
    ]
    assert result["benchmarks"][0]["delta_score"] == 1.0
    assert result["weight_sync"]["baseline"]["reference"] == "Qwen/base"
    assert result["weight_sync"]["final"]["checkpoint"] == str(final_model)
