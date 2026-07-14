from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.cli import (
    build_parser,
    default_hf_job_secrets,
    main,
)
from posttrainarena.benchflow_pipeline.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"


def test_validate_emits_machine_readable_json(capsys) -> None:
    assert main(["validate", "--config", str(CONFIG)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"config": str(CONFIG.resolve()), "valid": True}


def test_plan_emits_machine_readable_json(capsys) -> None:
    assert main(["plan", "--config", str(CONFIG), "--run-name", "cli-test"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_name"] == "cli-test"
    assert payload["train_task_count"] == 15


def test_model_bridge_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "model-bridge",
            "--tokenizer",
            "Qwen/Qwen3-4B",
            "--tokenizer-revision",
            "a" * 40,
            "--max-tokens",
            "2048",
            "--max-context-tokens",
            "32768",
            "--max-sidecar-entries",
            "256",
            "--port",
            "9001",
        ]
    )

    assert args.command == "model-bridge"
    assert args.tokenizer == "Qwen/Qwen3-4B"
    assert args.api_key_env == "BENCHFLOW_PROVIDER_API_KEY"
    assert args.max_tokens == 2048
    assert args.max_context_tokens == 32768
    assert args.max_sidecar_entries == 256
    assert args.port == 9001


def test_submission_uploads_default_private() -> None:
    private_args = build_parser().parse_args(
        [
            "prepare-submission",
            "--entry",
            "entry",
            "--base-config",
            "config.toml",
            "--out",
            "out",
            "--dataset-repo",
            "org/data",
        ]
    )
    public_args = build_parser().parse_args(
        [
            "prepare-submission",
            "--entry",
            "entry",
            "--base-config",
            "config.toml",
            "--out",
            "out",
            "--dataset-repo",
            "org/data",
            "--public",
        ]
    )

    assert private_args.private is True
    assert public_args.private is False


def test_qwen_teacher_hf_job_secrets_are_derived_from_recipe() -> None:
    names = default_hf_job_secrets(ROOT / "configs/qwen3.5-9b-data-agent-full.toml")

    assert "OPENROUTER_API_KEY" in names
    assert "DAYTONA_API_KEY" not in names
    assert "QWEN_API_KEY" not in names
    assert "QWEN_BASE_URL" not in names
    assert "GLM_API_KEY" not in names
    assert "WANDB_API_KEY" not in names


def test_hf_job_secrets_omit_disabled_services(monkeypatch) -> None:
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-full.toml")
    config = replace(
        config,
        runtime=replace(config.runtime, sandbox="docker"),
        teacher=replace(config.teacher, enabled=False),
        tracking=replace(config.tracking, report_to="none"),
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.cli.load_config",
        lambda _path: config,
    )

    names = default_hf_job_secrets(Path("unused.toml"))

    assert "DAYTONA_API_KEY" not in names
    assert "OPENROUTER_API_KEY" not in names
    assert "WANDB_API_KEY" not in names


def test_hf_job_secrets_require_known_teacher_provider(monkeypatch) -> None:
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-full.toml")
    config = replace(
        config,
        teacher=replace(config.teacher, model="unknown/model"),
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.cli.load_config",
        lambda _path: config,
    )

    with pytest.raises(ValueError, match="pass --secret-env explicitly"):
        default_hf_job_secrets(Path("unused.toml"))
