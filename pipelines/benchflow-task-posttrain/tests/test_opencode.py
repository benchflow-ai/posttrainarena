from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.io import CommandRunner
from posttrainarena.benchflow_pipeline.opencode import (
    build_evaluation_command,
    evaluate,
    evaluation_env,
    load_summary,
    opencode_config_env,
    served_model,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _served_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHFLOW_BASE_MODEL", "vllm/base")
    monkeypatch.setenv("BENCHFLOW_ADAPTER_MODEL", "vllm/student")
    monkeypatch.setenv(
        "BENCHFLOW_PROVIDER_BASE_URL",
        "https://example.invalid/v1",
    )
    monkeypatch.setenv("BENCHFLOW_PROVIDER_API_KEY", "secret")


class FakeRunner(CommandRunner):
    def __init__(self, cwd: Path, *, dry_run: bool = False) -> None:
        super().__init__(cwd=cwd, dry_run=dry_run)

    def run(
        self,
        name: str,
        command: list[str],
        *,
        check: bool = True,
        env_overrides: dict[str, str] | None = None,
    ) -> int:
        self.commands.append(
            {
                "name": name,
                "command": command,
                "env_keys": sorted(env_overrides or {}),
            }
        )
        return 0


def _config():
    return load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")


def _write_health(path: Path, **patch: Any) -> None:
    path.write_text(
        json.dumps(
            {
                "total_rows": 1,
                "scored_rows": 1,
                "unscored_rows": 0,
                "rows_with_tool_calls": 1,
                "zero_tool_rows": 0,
                "missing_llm_trajectory": 0,
                "malformed_llm_trajectory": 0,
                **patch,
            }
        )
    )


def test_served_model_distinguishes_base_and_trained_checkpoint() -> None:
    config = _config()

    assert served_model(config, config.model) == "vllm/base"
    assert served_model(config, "/tmp/sft") == "vllm/student"
    assert served_model(config, config.model, role="student") == "vllm/student"


def test_served_model_requires_provider_qualified_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHFLOW_BASE_MODEL", "bare-model")

    with pytest.raises(RuntimeError, match="provider/model format"):
        served_model(_config(), _config().model)


def test_evaluation_env_maps_named_secrets_without_exposing_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config = replace(
        config,
        evaluation=replace(
            config.evaluation,
            base_url_env="TEST_MODEL_URL",
            api_key_env="TEST_MODEL_KEY",
        ),
    )
    monkeypatch.setenv("TEST_MODEL_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")

    assert evaluation_env(config) == {
        "BENCHFLOW_PROVIDER_BASE_URL": "https://example.invalid/v1",
        "BENCHFLOW_PROVIDER_API_KEY": "secret",
    }


def test_evaluation_env_fails_closed_on_missing_value() -> None:
    config = _config()
    config = replace(
        config,
        evaluation=replace(config.evaluation, base_url_env="MISSING_MODEL_URL"),
    )

    with pytest.raises(RuntimeError, match="MISSING_MODEL_URL"):
        evaluation_env(config)


def test_build_evaluation_command_uses_opencode_contract(tmp_path: Path) -> None:
    config = _config()
    command = build_evaluation_command(
        config=config,
        model=config.model,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a", "task-b"],
        jobs_dir=tmp_path / "jobs",
        health_path=tmp_path / "health.json",
        task_manifest_path=tmp_path / "task-manifest.json",
        run_config_path=tmp_path / "run-config.json",
    )

    assert command[command.index("--agent") + 1] == "opencode"
    assert command[command.index("--model") + 1] == "vllm/base"
    assert command[command.index("--usage-tracking") + 1] == "required"
    assert command[command.index("--expected-tasks") + 1] == "2"
    assert command.count("--include") == 2
    assert opencode_config_env(config) in command
    inline = json.loads(opencode_config_env(config).split("=", 1)[1])
    assert inline["permission"]["external_directory"] == {
        "/home/user/input/**": "allow"
    }
    assert inline["permission"]["bash"] == {"*<<*": "deny"}


def test_build_evaluation_command_can_capture_sampled_token_logprobs(
    tmp_path: Path,
) -> None:
    command = build_evaluation_command(
        config=_config(),
        model="/tmp/student",
        model_role="student",
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a"],
        jobs_dir=tmp_path / "jobs",
        health_path=tmp_path / "health.json",
        task_manifest_path=tmp_path / "task-manifest.json",
        run_config_path=tmp_path / "run-config.json",
        capture_token_logprobs=True,
    )

    values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--agent-env"
    ]
    assert opencode_config_env(_config()) in values
    assert "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1" in values
    assert command[command.index("--model") + 1] == "vllm/student"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"total": 2}, "expected 1"),
        ({"errored": 1}, "agent or verifier errors"),
        ({"verifier_errored": 1}, "agent or verifier errors"),
        ({"telemetry_coverage": 0.5}, "telemetry coverage is incomplete"),
        ({"telemetry_coverage": float("nan")}, "telemetry_coverage is invalid"),
        ({"score_excl_errors_ratio": 1.1}, "score_excl_errors_ratio is invalid"),
    ],
)
def test_load_summary_fails_closed(
    tmp_path: Path,
    patch: dict[str, Any],
    message: str,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    summary = {
        "total": 1,
        "errored": 0,
        "verifier_errored": 0,
        "telemetry_coverage": 1.0,
        "score_excl_errors_ratio": 1.0,
        **patch,
    }
    (jobs_dir / "summary.json").write_text(json.dumps(summary))
    health_path = tmp_path / "health.json"
    _write_health(health_path)

    with pytest.raises(RuntimeError, match=message):
        load_summary(
            jobs_dir=jobs_dir,
            health_path=health_path,
            expected_tasks=1,
        )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"total_rows": 2}, "expected 1"),
        ({"scored_rows": 0}, "scored_rows=0"),
        ({"unscored_rows": 1}, "unscored_rows=1"),
        ({"missing_llm_trajectory": 1}, "missing_llm_trajectory=1"),
        ({"malformed_llm_trajectory": 1}, "malformed_llm_trajectory=1"),
    ],
)
def test_load_summary_rejects_unhealthy_artifacts(
    tmp_path: Path,
    patch: dict[str, Any],
    message: str,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "errored": 0,
                "verifier_errored": 0,
                "telemetry_coverage": 1.0,
                "score_excl_errors_ratio": 1.0,
            }
        )
    )
    health_path = tmp_path / "health.json"
    _write_health(health_path, **patch)

    with pytest.raises(RuntimeError, match=message):
        load_summary(
            jobs_dir=jobs_dir,
            health_path=health_path,
            expected_tasks=1,
        )


def test_load_summary_accepts_scored_zero_tool_failure(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "errored": 0,
                "verifier_errored": 0,
                "telemetry_coverage": 1.0,
                "score_excl_errors_ratio": 0.0,
            }
        )
    )
    health_path = tmp_path / "health.json"
    _write_health(
        health_path,
        rows_with_tool_calls=0,
        zero_tool_rows=1,
    )

    payload = load_summary(
        jobs_dir=jobs_dir,
        health_path=health_path,
        expected_tasks=1,
    )

    assert payload["score"] == 0.0
    assert payload["health"]["zero_tool_rows"] == 1


def test_load_summary_accepts_failed_attempt_followed_by_scored_retry(
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 2,
                "errored": 0,
                "verifier_errored": 0,
                "telemetry_coverage": 1.0,
                "score_excl_errors_ratio": 0.5,
            }
        )
    )
    health_path = tmp_path / "health.json"
    _write_health(
        health_path,
        total_rows=3,
        scored_rows=2,
        unscored_rows=1,
        rows=[
            {
                "task_id": "task-a",
                "scored": True,
                "error": None,
                "verifier_error": None,
                "valid_llm_trajectory": True,
            },
            {
                "task_id": "task-b",
                "scored": False,
                "error": "process exited",
                "verifier_error": None,
                "valid_llm_trajectory": True,
            },
            {
                "task_id": "task-b",
                "scored": True,
                "error": None,
                "verifier_error": None,
                "valid_llm_trajectory": True,
            },
        ],
    )

    payload = load_summary(
        jobs_dir=jobs_dir,
        health_path=health_path,
        expected_tasks=2,
        expected_task_ids=["task-a", "task-b"],
    )

    assert payload["score"] == 0.5
    assert payload["health"]["total_rows"] == 3


def test_evaluate_dry_run_records_only_environment_key_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "BENCHFLOW_BASE_MODEL",
        "BENCHFLOW_ADAPTER_MODEL",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
    ):
        monkeypatch.delenv(name)
    config = _config()
    runner = FakeRunner(ROOT, dry_run=True)

    payload = evaluate(
        config=config,
        runner=runner,
        stage="baseline_eval",
        model=config.model,
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a"],
        jobs_dir=tmp_path / "jobs",
        metrics_path=tmp_path / "metrics.json",
    )

    assert payload["served_model"] == "${BENCHFLOW_BASE_MODEL}"
    assert payload["score"] is None
    assert runner.commands[0]["env_keys"] == [
        "BENCHFLOW_PROVIDER_API_KEY",
        "BENCHFLOW_PROVIDER_BASE_URL",
    ]


def test_evaluate_writes_metrics_from_healthy_summary(tmp_path: Path) -> None:
    config = _config()
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "summary.json").write_text(
        json.dumps(
            {
                "total": 1,
                "errored": 0,
                "verifier_errored": 0,
                "telemetry_coverage": 1.0,
                "score_excl_errors_ratio": 0.5,
            }
        )
    )
    runner = FakeRunner(ROOT)
    metrics_path = tmp_path / "metrics.json"
    _write_health(
        tmp_path / "metrics_health.json",
        rows=[
            {
                "task_id": "task-a",
                "scored": True,
                "error": None,
                "verifier_error": None,
                "valid_llm_trajectory": True,
            }
        ],
    )

    payload = evaluate(
        config=config,
        runner=runner,
        stage="sft_eval",
        model="/tmp/sft",
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a"],
        jobs_dir=jobs_dir,
        metrics_path=metrics_path,
    )

    assert payload["score"] == 0.5
    assert payload["served_model"] == "vllm/student"
    assert json.loads(metrics_path.read_text())["harness"] == "opencode"
