from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.io import CommandRunner
from posttrainarena.benchflow_pipeline.teacher import (
    _select_verified,
    _training_ready,
    _valid_jsonl,
    build_teacher_command,
    collect_verified_teacher_rollouts,
)
from posttrainarena.benchflow_pipeline.opencode import opencode_config_env


ROOT = Path(__file__).resolve().parents[1]


def _write_training_ready_result(rollout: Path, *, total_tokens: int = 10) -> None:
    (rollout / "results.jsonl").write_text(
        json.dumps(
            {
                "info": {"training_ready": True},
                "token_usage": {"total_tokens": total_tokens},
                "is_completed": True,
                "is_truncated": False,
                "stop_condition": "agent_completed",
                "error": None,
            }
        )
        + "\n"
    )


def _write_complete_acp_trajectory(rollout: Path) -> None:
    trajectory = rollout / "trajectory"
    trajectory.mkdir(exist_ok=True)
    (trajectory / "acp_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "user_message", "text": "task"},
                {
                    "type": "tool_call",
                    "tool_call_id": "call-1",
                    "title": "bash",
                    "kind": "execute",
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": "ok"},
                        }
                    ],
                },
                {"type": "agent_message", "text": "done"},
            )
        )
        + "\n"
    )


class FakeRunner(CommandRunner):
    def __init__(
        self,
        cwd: Path,
        *,
        dry_run: bool,
        returncodes: list[int] | None = None,
    ) -> None:
        super().__init__(cwd=cwd, dry_run=dry_run)
        self.returncodes = iter(returncodes or [])

    def run(self, name: str, command: list[str], *, check: bool = True) -> int:
        self.commands.append({"name": name, "command": command})
        return next(self.returncodes, 0)


def test_build_teacher_command_uses_opencode_and_required_telemetry(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    command = build_teacher_command(
        config=config,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a", "task-b"],
        jobs_dir=tmp_path / "jobs",
        task_manifest_path=tmp_path / "task-manifest.json",
        run_config_path=tmp_path / "run-config.json",
        health_path=tmp_path / "health.json",
    )

    assert command[command.index("--agent") + 1] == "opencode"
    assert command[command.index("--model") + 1] == "glm/glm-5.1"
    assert command[command.index("--usage-tracking") + 1] == "required"
    assert command[command.index("--expected-tasks") + 1] == "2"
    assert (
        command[command.index("--config-override") + 1]
        == '{"agent":{"timeout_sec":900}}'
    )
    assert command.count("--include") == 2
    assert opencode_config_env(config) in command
    assert "--matrix" not in command
    assert "--trials" not in command


def test_build_teacher_command_rejects_empty_task_list(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")

    with pytest.raises(ValueError, match="at least one task"):
        build_teacher_command(
            config=config,
            tasks_dir=tmp_path / "tasks",
            task_ids=[],
            jobs_dir=tmp_path / "jobs",
            task_manifest_path=tmp_path / "task-manifest.json",
            run_config_path=tmp_path / "run-config.json",
            health_path=tmp_path / "health.json",
        )


def test_build_teacher_command_supports_root_user_and_reasoning_effort(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        runtime=replace(config.runtime, sandbox_user=None),
        harness=replace(config.harness, reasoning_effort="high"),
    )

    command = build_teacher_command(
        config=config,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a"],
        jobs_dir=tmp_path / "jobs",
        task_manifest_path=tmp_path / "task-manifest.json",
        run_config_path=tmp_path / "run-config.json",
        health_path=tmp_path / "health.json",
    )

    assert "--sandbox-user" not in command
    assert command[command.index("--reasoning-effort") + 1] == "high"


def test_training_ready_requires_one_valid_row(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    assert _training_ready(rollout) is False

    _write_training_ready_result(rollout)
    assert _training_ready(rollout) is True

    (rollout / "results.jsonl").write_text(
        json.dumps(
            {
                "info": {"training_ready": True},
                "token_usage": {"total_tokens": 10},
                "is_completed": True,
                "is_truncated": False,
                "stop_condition": "agent_completed",
                "error": None,
            }
        )
        + "\n"
        + json.dumps(
            {
                "info": {"training_ready": True},
                "token_usage": {"total_tokens": 10},
                "is_completed": True,
                "is_truncated": False,
                "stop_condition": "agent_completed",
                "error": None,
            }
        )
        + "\n"
    )
    assert _training_ready(rollout) is False


def test_training_ready_requires_positive_usage(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / "results.jsonl").write_text(
        json.dumps(
            {
                "info": {"training_ready": True},
                "token_usage": {"total_tokens": 0},
                "is_completed": True,
                "is_truncated": False,
                "stop_condition": "agent_completed",
                "error": None,
            }
        )
        + "\n"
    )
    assert _training_ready(rollout) is False

    (rollout / "results.jsonl").write_text(
        json.dumps(
            {
                "info": {"training_ready": True},
                "token_usage": {"input_tokens": 7, "output_tokens": 3},
                "is_completed": True,
                "is_truncated": False,
                "stop_condition": "agent_completed",
                "error": None,
            }
        )
        + "\n"
    )
    assert _training_ready(rollout) is True


def test_valid_jsonl_requires_parseable_nonempty_file(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    assert _valid_jsonl(path) is False
    path.write_text("")
    assert _valid_jsonl(path) is False
    path.write_text("{bad json}\n")
    assert _valid_jsonl(path) is False
    path.write_text("null\n")
    assert _valid_jsonl(path) is False
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "user_message", "text": "task"},
                {
                    "type": "tool_call",
                    "tool_call_id": "call-1",
                    "title": "bash",
                    "kind": "execute",
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": "ok"},
                        }
                    ],
                },
                {"type": "agent_message", "text": "done"},
            )
        )
        + "\n"
    )
    assert _valid_jsonl(path) is True

    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "user_message", "text": "task"},
                {"type": "tool_call", "tool_call_id": ""},
                {"type": "agent_message", "text": "done"},
            )
        )
        + "\n"
    )
    assert _valid_jsonl(path) is False

    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "user_message", "text": "task"},
                {
                    "type": "tool_call",
                    "tool_call_id": "call-1",
                    "title": "bash",
                    "kind": "execute",
                    "status": "completed",
                    "content": [{"type": "content"}],
                },
                {"type": "agent_message", "text": "done"},
            )
        )
        + "\n"
    )
    assert _valid_jsonl(path) is False


def test_select_verified_filters_unhealthy_and_picks_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    first = tmp_path / "task-a__first"
    second = tmp_path / "task-a__second"
    low = tmp_path / "task-b__low"
    for rollout in (first, second, low):
        rollout.mkdir()
        _write_training_ready_result(rollout)
        _write_complete_acp_trajectory(rollout)
    rows = [
        {
            "task_id": "task-a",
            "rollout_dir": str(first),
            "reward": 1.0,
            "scored": True,
            "tool_calls": 2,
            "valid_llm_trajectory": True,
            "llm_trajectory_rows": 3,
            "error": None,
            "verifier_error": None,
        },
        {
            "task_id": "task-a",
            "rollout_dir": str(second),
            "reward": 1.0,
            "scored": True,
            "tool_calls": 4,
            "valid_llm_trajectory": True,
            "llm_trajectory_rows": 5,
            "error": None,
            "verifier_error": None,
        },
        {
            "task_id": "task-b",
            "rollout_dir": str(low),
            "reward": 0.5,
            "scored": True,
            "tool_calls": 3,
            "valid_llm_trajectory": True,
            "llm_trajectory_rows": 4,
            "error": None,
            "verifier_error": None,
        },
    ]
    monkeypatch.setattr(
        "benchflow.eval_artifacts.build_health_summary",
        lambda _jobs_dir: {"rows": rows},
    )

    attempts, selected = _select_verified(
        config=config,
        jobs_dir=tmp_path,
        task_ids=["task-a", "task-b"],
    )

    assert len(attempts) == 3
    assert [row["task_id"] for row in selected] == ["task-a"]
    assert selected[0]["rollout_dir"] == str(second)
    assert selected[0]["training_ready"] is True
    assert selected[0]["acp_trajectory_ready"] is True


def test_select_verified_enforces_token_and_tool_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout = tmp_path / "task-a__expensive"
    rollout.mkdir()
    _write_training_ready_result(rollout, total_tokens=100)
    _write_complete_acp_trajectory(rollout)
    monkeypatch.setattr(
        "benchflow.eval_artifacts.build_health_summary",
        lambda _jobs_dir: {
            "rows": [
                {
                    "task_id": "task-a",
                    "rollout_dir": str(rollout),
                    "reward": 1.0,
                    "scored": True,
                    "tool_calls": 10,
                    "valid_llm_trajectory": True,
                    "llm_trajectory_rows": 3,
                    "error": None,
                    "verifier_error": None,
                }
            ]
        },
    )
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            max_accepted_total_tokens=50,
            max_accepted_tool_calls=5,
        ),
    )

    attempts, selected = _select_verified(
        config=config,
        jobs_dir=tmp_path,
        task_ids=["task-a"],
    )

    assert attempts[0]["total_tokens"] == 100
    assert attempts[0]["eligible"] is False
    assert selected == []


def test_teacher_config_accepts_fractional_reward_threshold() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, teacher=replace(config.teacher, min_reward=0.5))

    config.validate()


def test_collect_teacher_dry_run_records_all_possible_attempts(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(config.teacher, max_attempts=2, min_verified=1),
    )
    runner = FakeRunner(ROOT, dry_run=True)

    manifest = collect_verified_teacher_rollouts(
        config=config,
        runner=runner,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a", "task-b"],
        jobs_dir=tmp_path / "jobs",
        manifest_path=tmp_path / "reports" / "teacher.json",
        selection_path=tmp_path / "reports" / "selection.json",
    )

    assert manifest["verified_count"] is None
    assert manifest["teacher_source_identity_enforced"] is False
    assert [record["name"] for record in runner.commands] == [
        "collect_verified_teacher_rollouts",
        "collect_verified_teacher_rollouts_attempt-02",
    ]
    assert all(record["command"].count("--include") == 2 for record in runner.commands)


def test_collect_teacher_retries_only_tasks_without_verified_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(config.teacher, max_attempts=3, min_verified=2),
    )
    runner = FakeRunner(ROOT, dry_run=False)
    task_a = {
        "task_id": "task-a",
        "rollout_dir": str(tmp_path / "task-a"),
        "reward": 1.0,
        "eligible": True,
    }
    task_b = {
        "task_id": "task-b",
        "rollout_dir": str(tmp_path / "task-b"),
        "reward": 1.0,
        "eligible": True,
    }
    responses = iter(
        [
            ([task_a], [task_a]),
            ([task_b], [task_b]),
        ]
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.teacher._select_verified",
        lambda **_kwargs: next(responses),
    )

    manifest = collect_verified_teacher_rollouts(
        config=config,
        runner=runner,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a", "task-b"],
        jobs_dir=tmp_path / "jobs",
        manifest_path=tmp_path / "reports" / "teacher.json",
        selection_path=tmp_path / "reports" / "selection.json",
    )

    assert manifest["verified_count"] == 2
    assert len(runner.commands) == 2
    assert runner.commands[0]["command"].count("--include") == 2
    second_command = runner.commands[1]["command"]
    assert second_command.count("--include") == 1
    assert "task-b" in second_command
    assert "task-a" not in second_command
    selection = json.loads((tmp_path / "reports" / "selection.json").read_text())
    assert selection["selected_count"] == 2


def test_collect_teacher_retries_after_rollout_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(config.teacher, max_attempts=2, min_verified=1),
    )
    runner = FakeRunner(ROOT, dry_run=False, returncodes=[1, 0])
    selected = {
        "task_id": "task-a",
        "rollout_dir": str(tmp_path / "task-a"),
        "reward": 1.0,
        "eligible": True,
    }
    responses = iter([([], []), ([selected], [selected])])
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.teacher._select_verified",
        lambda **_kwargs: next(responses),
    )

    manifest = collect_verified_teacher_rollouts(
        config=config,
        runner=runner,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a"],
        jobs_dir=tmp_path / "jobs",
        manifest_path=tmp_path / "reports" / "teacher.json",
        selection_path=tmp_path / "reports" / "selection.json",
    )

    assert manifest["verified_count"] == 1
    assert len(runner.commands) == 2


def test_collect_teacher_rejects_cli_usage_error(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    runner = FakeRunner(ROOT, dry_run=False, returncodes=[2])

    with pytest.raises(RuntimeError, match="exit 2"):
        collect_verified_teacher_rollouts(
            config=config,
            runner=runner,
            tasks_dir=tmp_path / "tasks",
            task_ids=["task-a"],
            jobs_dir=tmp_path / "jobs",
            manifest_path=tmp_path / "reports" / "teacher.json",
            selection_path=tmp_path / "reports" / "selection.json",
        )


def test_collect_teacher_requires_one_verified_rollout_per_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(config.teacher, max_attempts=1, min_verified=1),
    )
    runner = FakeRunner(ROOT, dry_run=False, returncodes=[0])
    selected = {
        "task_id": "task-a",
        "rollout_dir": str(tmp_path / "task-a"),
        "reward": 1.0,
        "eligible": True,
    }
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.teacher._select_verified",
        lambda **_kwargs: ([selected], [selected]),
    )

    with pytest.raises(RuntimeError, match="required 2"):
        collect_verified_teacher_rollouts(
            config=config,
            runner=runner,
            tasks_dir=tmp_path / "tasks",
            task_ids=["task-a", "task-b"],
            jobs_dir=tmp_path / "jobs",
            manifest_path=tmp_path / "reports" / "teacher.json",
            selection_path=tmp_path / "reports" / "selection.json",
        )


def test_collect_teacher_can_use_minimum_when_all_tasks_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            max_attempts=1,
            min_verified=1,
            require_all_tasks=False,
        ),
    )
    runner = FakeRunner(ROOT, dry_run=False, returncodes=[0])
    selected = {
        "task_id": "task-a",
        "rollout_dir": str(tmp_path / "task-a"),
        "reward": 1.0,
        "eligible": True,
    }
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.teacher._select_verified",
        lambda **_kwargs: ([selected], [selected]),
    )

    manifest = collect_verified_teacher_rollouts(
        config=config,
        runner=runner,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a", "task-b"],
        jobs_dir=tmp_path / "jobs",
        manifest_path=tmp_path / "reports" / "teacher.json",
        selection_path=tmp_path / "reports" / "selection.json",
    )

    assert manifest["required_verified_count"] == 1
    assert manifest["verified_count"] == 1


def test_collect_teacher_does_not_retry_over_budget_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            max_attempts=3,
            min_verified=1,
            max_accepted_total_tokens=50,
        ),
    )
    runner = FakeRunner(ROOT, dry_run=False, returncodes=[0])
    attempt = {
        "task_id": "task-a",
        "rollout_dir": str(tmp_path / "task-a"),
        "reward": 1.0,
        "tool_calls": 2,
        "total_tokens": 100,
        "eligible": False,
    }
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.teacher._select_verified",
        lambda **_kwargs: ([attempt], []),
    )

    with pytest.raises(RuntimeError, match="Only 0 verified trajectories"):
        collect_verified_teacher_rollouts(
            config=config,
            runner=runner,
            tasks_dir=tmp_path / "tasks",
            task_ids=["task-a"],
            jobs_dir=tmp_path / "jobs",
            manifest_path=tmp_path / "reports" / "teacher.json",
            selection_path=tmp_path / "reports" / "selection.json",
        )

    assert len(runner.commands) == 1
