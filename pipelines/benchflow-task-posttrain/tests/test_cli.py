from __future__ import annotations

import json
from pathlib import Path

from posttrainarena.benchflow_pipeline.cli import build_parser, main


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
            "--port",
            "9001",
        ]
    )

    assert args.command == "model-bridge"
    assert args.tokenizer == "Qwen/Qwen3-4B"
    assert args.api_key_env == "BENCHFLOW_PROVIDER_API_KEY"
    assert args.max_tokens == 2048
    assert args.port == 9001
