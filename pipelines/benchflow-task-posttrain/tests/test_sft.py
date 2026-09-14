from __future__ import annotations

import json
import sys
from collections import UserDict
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.checkpoint import export_merged_checkpoint
from posttrainarena.benchflow_pipeline.sft import (
    build_tokenized_sft_rows,
    _token_ids,
    load_trl_rows,
    train_sft,
    train_sft_adapter,
)


ROOT = Path(__file__).resolve().parents[1]


def test_token_ids_accepts_batch_encoding_mapping() -> None:
    assert _token_ids(UserDict({"input_ids": [[1, 2, 3]]})) == [1, 2, 3]


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
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            tools,
            add_generation_prompt=False,
        ):
            assert tokenize is True
            assert isinstance(tools, list)
            text = "".join(
                f"{message['role']}:{message.get('content') or ''}|"
                for message in messages
            )
            if add_generation_prompt:
                text += "assistant:"
            return list(text.encode())

        def save_pretrained(self, path):
            captured.setdefault("tokenizer_paths", []).append(path)
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    class FakeModel:
        pass

    class FakeMerged:
        def save_pretrained(self, path, **kwargs):
            captured["merged_save"] = (path, kwargs)
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"merged")

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
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")

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
    assert trainer_kwargs["train_dataset"][0]["input_ids"]
    assert -100 in trainer_kwargs["train_dataset"][0]["labels"]
    assert any(label != -100 for label in trainer_kwargs["train_dataset"][0]["labels"])
    assert args["completion_only_loss"] is False
    assert args["assistant_only_loss"] is False
    assert args["max_steps"] == config.sft.max_steps
    assert "num_train_epochs" not in args
    assert args["gradient_checkpointing"] is True
    assert trainer_kwargs["peft_config"].values["lora_dropout"] == (
        config.sft.lora_dropout
    )
    assert "dataset_text_field" not in args
    dependency = json.loads((tmp_path / "adapter/adapter_dependency.json").read_text())
    assert dependency["base_model"] == config.model
    assert dependency["base_revision"] == config.model_revision
    metrics = json.loads((tmp_path / "merged/train_metrics.json").read_text())
    assert metrics["train_jsonl_sha256"]
    assert metrics["adapter_sha256"]
    assert metrics["merged_model_sha256"]


def test_full_recipe_sft_uses_one_epoch_instead_of_max_steps(
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
        '"tools":[]}\n'
    )
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-full.toml")
    captured: dict[str, Any] = {}

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            tools,
            add_generation_prompt=False,
        ):
            assert tokenize is True
            assert isinstance(tools, list)
            text = "".join(
                f"{message['role']}:{message.get('content') or ''}|"
                for message in messages
            )
            if add_generation_prompt:
                text += "assistant:"
            return list(text.encode())

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    class FakeModel:
        pass

    class FakeMerged:
        def save_pretrained(self, path, **_kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"merged")

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, _model, _path):
            return cls()

        def merge_and_unload(self):
            return FakeMerged()

    class FakeConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["args"] = kwargs["args"].values

        def train(self):
            return SimpleNamespace(metrics={"loss": 0.1})

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_bytes(b"adapter")

    monkeypatch.setattr(
        datasets.Dataset,
        "from_list",
        lambda rows: rows,
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
    monkeypatch.setattr("peft.PeftModel", FakePeftModel)
    monkeypatch.setattr("peft.LoraConfig", FakeConfig)
    monkeypatch.setattr("trl.SFTConfig", FakeConfig)
    monkeypatch.setattr("trl.SFTTrainer", FakeTrainer)
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
        run_name="full",
    )

    assert captured["args"]["num_train_epochs"] == 1.0
    assert "max_steps" not in captured["args"]
    assert json.loads((tmp_path / "merged/train_metrics.json").read_text())[
        "merged_model_sha256"
    ]


def test_tokenized_sft_labels_start_at_exact_common_prefix() -> None:
    class PrefixMismatchTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            tools,
            add_generation_prompt=False,
        ):
            del tokenize, tools
            if add_generation_prompt:
                return [1, 2, 9]
            assert len(messages) == 2
            return [1, 2, 8, 3, 4]

    rows, stats = build_tokenized_sft_rows(
        [
            {
                "prompt": [{"role": "user", "content": "solve"}],
                "completion": [{"role": "assistant", "content": "done"}],
                "tools": [],
            }
        ],
        PrefixMismatchTokenizer(),
        max_length=16,
    )

    assert rows == [
        {
            "input_ids": [1, 2, 8, 3, 4],
            "labels": [-100, -100, 8, 3, 4],
        }
    ]
    assert stats == {
        "max_prompt_prefix_mismatch": 1,
        "trained_tokens": 3,
    }


def test_sft_adapter_is_published_by_the_main_process_only(
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
        '"tools":[]}\n'
    )
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    saves: list[str] = []
    barriers: list[str] = []

    class FakeTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, tools, add_generation_prompt=False
        ):
            del tokenize, tools
            return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4]

        def save_pretrained(self, path):
            saves.append(path)

    class FakeConfig:
        def __init__(self, **kwargs):
            self.values = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.accelerator = SimpleNamespace(
                is_main_process=False,
                wait_for_everyone=lambda: barriers.append("barrier"),
            )

        def train(self):
            return SimpleNamespace(metrics={"loss": 0.1})

        def save_model(self, path):
            saves.append(f"collective:{path}")

    monkeypatch.setattr(datasets.Dataset, "from_list", lambda rows: rows)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr("peft.LoraConfig", FakeConfig)
    monkeypatch.setattr("trl.SFTConfig", FakeConfig)
    monkeypatch.setattr("trl.SFTTrainer", FakeTrainer)
    monkeypatch.setattr(sft_module, "supported_kwargs", lambda _c, values: values)

    metrics = train_sft_adapter(
        config=config,
        train_jsonl=source,
        adapter_dir=tmp_path / "adapter",
        run_name="follower",
    )

    assert saves == [f"collective:{tmp_path / 'adapter'}"]
    assert barriers == ["barrier"]
    assert metrics["launch"] is None
    assert metrics["adapter_dir"] == str(tmp_path / "adapter")
    assert not (tmp_path / "adapter" / "train_metrics.json").exists()


def test_export_merges_on_cpu_and_completes_provisional_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter_dir / "train_metrics.json").write_text(
        json.dumps({"mode": "sft", "adapter_dir": str(adapter_dir)})
    )
    loads: list[dict[str, Any]] = []

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    class FakeMerged:
        def save_pretrained(self, path, **kwargs):
            assert kwargs == {"safe_serialization": True}
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"merged")

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path):
            assert path == str(adapter_dir)
            return cls()

        def merge_and_unload(self):
            return FakeMerged()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda model, **kwargs: loads.append({"model": model, **kwargs}) or object(),
    )
    monkeypatch.setattr("peft.PeftModel", FakePeftModel)

    payload = export_merged_checkpoint(
        base_model="Qwen/Qwen3-4B",
        base_kwargs={"trust_remote_code": True, "revision": "abc"},
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "merged",
    )

    assert loads == [
        {
            "model": "Qwen/Qwen3-4B",
            "dtype": "bfloat16",
            "device_map": "cpu",
            "trust_remote_code": True,
            "revision": "abc",
        }
    ]
    assert payload["mode"] == "sft"
    assert payload["merged_model_dir"] == str(tmp_path / "merged")
    assert payload["adapter_sha256"] and payload["merged_model_sha256"]
    on_disk = json.loads((tmp_path / "merged" / "train_metrics.json").read_text())
    assert on_disk == payload
