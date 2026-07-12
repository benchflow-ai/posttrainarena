from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.sft import load_trl_rows
from posttrainarena.benchflow_pipeline.sft import train_sft


ROOT = Path(__file__).resolve().parents[1]


def test_load_trl_rows_preserves_tools_and_object_arguments(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"prompt":[{"role":"user","content":"solve"},'
        '{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function",'
        '"function":{"name":"submit","arguments":{"answer":"done"}}}]},'
        '{"role":"tool","tool_call_id":"call-1","content":"reward=1"}],'
        '"completion":[{"role":"assistant","content":"done"}],'
        '"tools":[{"type":"function","function":{"name":"submit",'
        '"parameters":{"type":"object"}}}]}\n'
    )

    rows = load_trl_rows(source)

    assert rows[0]["completion"] == [{"role": "assistant", "content": "done"}]
    arguments = rows[0]["prompt"][1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"answer": "done"}
    assert rows[0]["tools"][0]["function"]["name"] == "submit"


def test_train_sft_uses_native_trl_rows_and_masked_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets
    import transformers
    import posttrainarena.benchflow_pipeline.sft as sft_module

    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"prompt":[{"role":"user","content":"solve"}],'
        '"completion":[{"role":"assistant","content":"done"}],'
        '"tools":[{"type":"function","function":{"name":"submit",'
        '"parameters":{"type":"object"}}}]}\n'
    )
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    captured: dict[str, Any] = {}

    class FakeTokenizer:
        def save_pretrained(self, path):
            captured.setdefault("tokenizer_paths", []).append(path)

    class FakeModel:
        pass

    class FakeMerged:
        def save_pretrained(self, path, **kwargs):
            captured["merged_save"] = (path, kwargs)

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path):
            captured["adapter_load"] = (model, path)
            return cls()

        def merge_and_unload(self):
            return FakeMerged()

    class FakeConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def train(self):
            return SimpleNamespace(metrics={"loss": 0.25})

        def save_model(self, path):
            captured["adapter_save"] = path

    fake_peft = ModuleType("peft")
    fake_peft.__spec__ = ModuleSpec("peft", loader=None)
    fake_peft.PeftModel = FakePeftModel  # type: ignore[attr-defined]
    fake_peft.LoraConfig = FakeConfig  # type: ignore[attr-defined]
    fake_trl = ModuleType("trl")
    fake_trl.__spec__ = ModuleSpec("trl", loader=None)
    fake_trl.SFTConfig = FakeConfig  # type: ignore[attr-defined]
    fake_trl.SFTTrainer = FakeTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setattr(
        datasets.Dataset,
        "from_list",
        lambda rows: captured.setdefault("dataset_rows", rows) or rows,
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        sft_module,
        "supported_kwargs",
        lambda _callable, values: values,
    )

    train_sft(
        config=config,
        train_jsonl=source,
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "merged",
        run_name="test",
    )

    trainer_kwargs = captured["trainer_kwargs"]
    args = trainer_kwargs["args"].values
    assert trainer_kwargs["train_dataset"] == captured["dataset_rows"]
    assert trainer_kwargs["train_dataset"][0]["prompt"][0]["content"] == "solve"
    assert trainer_kwargs["train_dataset"][0]["tools"][0]["function"]["name"] == (
        "submit"
    )
    assert args["completion_only_loss"] is True
    assert args["assistant_only_loss"] is True
    assert "dataset_text_field" not in args
