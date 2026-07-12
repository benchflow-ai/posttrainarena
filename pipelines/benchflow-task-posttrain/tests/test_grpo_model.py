from __future__ import annotations

from pathlib import Path

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.grpo import _model_init_kwargs


ROOT = Path(__file__).resolve().parents[1]


def test_grpo_model_loads_are_bfloat16_and_only_base_is_revision_pinned() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")

    base = _model_init_kwargs(config, config.model)
    merged = _model_init_kwargs(config, "/tmp/sft-merged")

    assert base["dtype"] == "bfloat16"
    assert base["revision"] == config.model_revision
    assert merged["dtype"] == "bfloat16"
    assert "revision" not in merged
