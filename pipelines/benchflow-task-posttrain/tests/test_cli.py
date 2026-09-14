from __future__ import annotations

import json
import sys
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
    assert payload == {
        "config": str(CONFIG.resolve()),
        "launch": {"sft": None, "grpo": None},
        "valid": True,
    }


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
            "--max-logprob-context-tokens",
            "12288",
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
    assert args.max_logprob_context_tokens == 12288
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


def test_worker_and_standalone_sft_commands_parse() -> None:
    parser = build_parser()

    worker = parser.parse_args(
        [
            "grpo-worker",
            "--config",
            str(CONFIG),
            "--model",
            "runs/x/checkpoints/sft-merged",
            "--tasks-dir",
            "runs/x/data/train",
            "--task-ids-file",
            "runs/x/jobs/grpo-train/task_ids.txt",
            "--jobs-dir",
            "runs/x/jobs/grpo-train",
            "--adapter-dir",
            "runs/x/checkpoints/grpo-adapter",
            "--run-name",
            "x-grpo",
        ]
    )
    standalone = parser.parse_args(
        [
            "sft",
            "--config",
            str(CONFIG),
            "--train-jsonl",
            "data/train.jsonl",
            "--adapter-dir",
            "out/adapter",
            "--output-dir",
            "out/merged",
            "--run-name",
            "offline",
        ]
    )

    assert worker.command == "grpo-worker"
    assert worker.task_ids_file == Path("runs/x/jobs/grpo-train/task_ids.txt")
    assert standalone.command == "sft"
    assert standalone.output_dir == Path("out/merged")
    with pytest.raises(SystemExit):
        parser.parse_args(["sft-worker", "--config", str(CONFIG)])


def test_standalone_sft_delegates_to_the_stage_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.launcher.run_sft_stage",
        lambda **kwargs: (
            calls.append(kwargs) or {"mode": "sft", "run": kwargs["run_name"]}
        ),
    )

    assert (
        main(
            [
                "sft",
                "--config",
                str(CONFIG),
                "--train-jsonl",
                str(tmp_path / "train.jsonl"),
                "--adapter-dir",
                str(tmp_path / "adapter"),
                "--output-dir",
                str(tmp_path / "merged"),
                "--run-name",
                "offline",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == {"mode": "sft", "run": "offline"}
    assert calls[0]["train_jsonl"] == tmp_path / "train.jsonl"
    assert calls[0]["runner"].cwd == CONFIG.resolve().parent


def test_standalone_sft_resolves_paths_before_changing_worker_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from posttrainarena.benchflow_pipeline import checkpoint, launcher

    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "train.jsonl").write_text("input from the caller directory")
    recipe = tmp_path / "recipes" / "sft.toml"
    recipe.parent.mkdir()
    profile = ROOT / "configs/accelerate/one-process.yaml"
    recipe.write_text(
        CONFIG.read_text()
        .replace('task_list = "../task-lists/', f'task_list = "{ROOT}/task-lists/')
        .replace("[sft]\n", f'[sft]\naccelerate_config = "{profile}"\n')
    )
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from posttrainarena.benchflow_pipeline.cli import build_parser\n"
        "args = build_parser().parse_args()\n"
        "data = Path(args.train_jsonl).read_text()\n"
        "args.adapter_dir.mkdir(parents=True)\n"
        "(args.adapter_dir / 'train_metrics.json').write_text(json.dumps(\n"
        "    {'input': data, 'cwd': str(Path.cwd())}))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setattr(
        launcher.LaunchProfile,
        "command",
        lambda self, name, arguments: [sys.executable, str(worker), name, *arguments],
    )

    def export(*, adapter_dir, output_dir, **kwargs):
        # Only the model export is replaced; the worker reads/writes real files.
        metrics = json.loads((adapter_dir / "train_metrics.json").read_text())
        output_dir.mkdir(parents=True)
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics))
        return metrics

    monkeypatch.setattr(checkpoint, "export_merged_checkpoint", export)
    monkeypatch.chdir(caller)
    assert (
        main(
            [
                "sft",
                "--config",
                str(recipe),
                "--train-jsonl",
                "train.jsonl",
                "--adapter-dir",
                "out/adapter",
                "--output-dir",
                "out/merged",
                "--run-name",
                "relative-paths",
            ]
        )
        == 0
    )

    metrics = json.loads(capsys.readouterr().out)
    assert metrics == {
        "input": "input from the caller directory",
        "cwd": str(recipe.parent),
    }
    assert (caller / "out/adapter/train_metrics.json").is_file()
    assert (caller / "out/merged/train_metrics.json").is_file()
    assert not (recipe.parent / "out").exists()
