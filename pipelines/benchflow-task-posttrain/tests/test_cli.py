from __future__ import annotations

import json
from pathlib import Path

from posttrainarena.benchflow_pipeline.cli import main


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
