from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.io import CommandRunner
from posttrainarena.benchflow_pipeline.launcher import (
    LaunchProfile,
    bundle_stage_profiles,
    run_grpo_stage,
    run_sft_stage,
    stage_summary,
    validate_grpo_topology,
    validate_profiles,
    verify_worker_topology,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "accelerate"
SMOKE = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"


def _with_profiles(config, *, sft: Path | None = None, grpo: Path | None = None):
    return replace(
        config,
        sft=replace(config.sft, accelerate_config=sft),
        grpo=replace(config.grpo, accelerate_config=grpo),
    )


def test_shipped_profiles_load_and_pick_the_matching_launcher() -> None:
    single = LaunchProfile.load(PROFILES / "one-process.yaml")
    ddp = LaunchProfile.load(PROFILES / "ddp-2gpu.yaml")

    assert single.num_processes == 1
    assert single.command("sft-worker", ["--x", "1"]) == [
        sys.executable,
        "-m",
        "posttrainarena.benchflow_pipeline.cli",
        "sft-worker",
        "--x",
        "1",
    ]
    assert ddp.distributed_type == "MULTI_GPU"
    assert ddp.command("grpo-worker", [])[:6] == [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(ddp.path),
        "-m",
    ]


DDP = "distributed_type: MULTI_GPU\nnum_processes: 2\nmixed_precision: bf16\n"


def test_profile_fingerprint_includes_additional_accelerate_settings(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(DDP)
    before = LaunchProfile.load(profile_path)
    profile_path.write_text(DDP + "enable_cpu_affinity: true\n")
    after = LaunchProfile.load(profile_path)

    assert after.fingerprint != before.fingerprint


def test_profile_fingerprint_ignores_deployment_only_keys(tmp_path: Path) -> None:
    shipped = LaunchProfile.load(PROFILES / "ddp-2gpu.yaml")
    moved = tmp_path / "moved.yaml"
    moved.write_text(
        (PROFILES / "ddp-2gpu.yaml")
        .read_text()
        .replace("gpu_ids: all", "gpu_ids: 2,3\nmain_process_port: 29777")
    )
    resharded = tmp_path / "four.yaml"
    resharded.write_text(
        (PROFILES / "ddp-2gpu.yaml")
        .read_text()
        .replace("num_processes: 2", "num_processes: 4")
    )

    terse = tmp_path / "terse.yaml"
    terse.write_text(DDP)

    assert LaunchProfile.load(moved).fingerprint == shipped.fingerprint
    assert LaunchProfile.load(terse).fingerprint == shipped.fingerprint
    assert LaunchProfile.load(resharded).fingerprint != shipped.fingerprint
    assert shipped.summary() == {
        "distributed_type": "MULTI_GPU",
        "num_processes": 2,
        "fingerprint": shipped.fingerprint,
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("num_processes: 2\nmixed_precision: bf16\n", "distributed_type is required"),
        (DDP.replace("MULTI_GPU", "DEEPSPEED"), "distributed_type must be one of"),
        (DDP.replace("num_processes: 2", "num_processes: 0"), "positive integer"),
        (DDP.replace("num_processes: 2", "num_processes: 1"), "need at least 2"),
        (DDP.replace("MULTI_GPU", '"NO"'), "NO needs num_processes 1"),
        (DDP + "num_machines: 2\n", "num_machines"),
        (DDP + "compute_environment: AMAZON_SAGEMAKER\n", "LOCAL_MACHINE"),
        (DDP.replace("bf16", "fp16"), "bf16"),
        (DDP + "use_cpu: true\n", "use_cpu"),
        (
            DDP + "parallelism_config:\n  parallelism_config_cp_size: 2\n",
            "parallelism_config",
        ),
        (
            DDP.replace("MULTI_GPU", "FSDP")
            + "fsdp_config:\n  fsdp_version: 1\n  fsdp_state_dict_type: FULL_STATE_DICT\n",
            "fsdp_version",
        ),
        (DDP.replace("MULTI_GPU", "FSDP"), "FULL_STATE_DICT"),
        (
            DDP.replace("MULTI_GPU", "FSDP")
            + "fsdp_config:\n  fsdp_state_dict_type: SHARDED_STATE_DICT\n",
            "FULL_STATE_DICT",
        ),
        ("- not\n- a\n- mapping\n", "not a mapping"),
    ],
)
def test_profile_rejects_unsupported_settings(
    tmp_path: Path, text: str, message: str
) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        LaunchProfile.load(path)


def test_fsdp_profile_needs_a_full_state_dict_and_launches_through_accelerate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fsdp.yaml"
    path.write_text(
        DDP.replace("MULTI_GPU", "FSDP")
        + "fsdp_config:\n  fsdp_version: 2\n  fsdp_state_dict_type: FULL_STATE_DICT\n"
    )

    profile = LaunchProfile.load(path)

    assert profile.distributed is True
    assert profile.command("sft-worker", [])[1:3] == [
        "-m",
        "accelerate.commands.launch",
    ]
    assert profile.manifest() == {
        **profile.summary(),
        "settings": {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP",
            "num_processes": 2,
            "num_machines": 1,
            "mixed_precision": "bf16",
            "use_cpu": False,
            "fsdp_config": {
                "fsdp_version": 2,
                "fsdp_state_dict_type": "FULL_STATE_DICT",
            },
        },
        "profile": str(path),
        "profile_sha256": profile.manifest()["profile_sha256"],
    }
    assert len(profile.manifest()["profile_sha256"]) == 64


def test_shipped_fsdp2_profile_is_valid_and_distributed() -> None:
    profile = LaunchProfile.load(PROFILES / "fsdp2-2gpu.yaml")

    assert profile.distributed_type == "FSDP"
    assert profile.num_processes == 2
    assert profile.settings["fsdp_config"]["fsdp_version"] == 2
    assert profile.settings["fsdp_config"]["fsdp_state_dict_type"] == "FULL_STATE_DICT"


def test_bundle_stage_profiles_copies_and_repoints_each_stage(tmp_path: Path) -> None:
    recipe_dir = tmp_path / "recipes"
    (recipe_dir / "accelerate").mkdir(parents=True)
    (recipe_dir / "accelerate" / "ddp.yaml").write_text(DDP)
    data = {
        "sft": {"accelerate_config": "accelerate/ddp.yaml"},
        "grpo": {"learning_rate": 1e-6},
    }
    bundle = tmp_path / "bundle"

    bundle_stage_profiles(data, source_dir=recipe_dir, output_dir=bundle)

    assert data["sft"]["accelerate_config"] == "accelerate/sft.yaml"
    assert "accelerate_config" not in data["grpo"]
    assert (bundle / "accelerate" / "sft.yaml").read_text() == DDP


def _ddp_profile(tmp_path: Path, processes: int) -> LaunchProfile:
    target = tmp_path / f"ddp-{processes}.yaml"
    target.write_text(
        (PROFILES / "ddp-2gpu.yaml")
        .read_text()
        .replace("num_processes: 2", f"num_processes: {processes}")
    )
    return LaunchProfile.load(target)


def test_grpo_topology_mirrors_trl_batch_and_rollout_budget(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    ddp = LaunchProfile.load(PROFILES / "ddp-2gpu.yaml")
    validate_grpo_topology(
        replace(config, harness=replace(config.harness, concurrency=2)),
        ddp,
    )

    with pytest.raises(ValueError, match="leaves no rollout slot"):
        validate_grpo_topology(config, ddp)
    six_rollouts = replace(
        config,
        runtime=replace(config.runtime, num_generations=3),
        harness=replace(config.harness, concurrency=4),
        grpo=replace(config.grpo, generation_batch_size=6),
    )
    with pytest.raises(ValueError, match="divisible by the 4 trainer"):
        validate_grpo_topology(six_rollouts, _ddp_profile(tmp_path, 4))


def test_validate_profiles_reports_both_stages(tmp_path: Path) -> None:
    config = load_config(SMOKE)
    assert validate_profiles(config) == {"sft": None, "grpo": None}

    staged = _with_profiles(
        replace(config, harness=replace(config.harness, concurrency=2)),
        sft=PROFILES / "one-process.yaml",
        grpo=PROFILES / "ddp-2gpu.yaml",
    )
    launch = validate_profiles(staged)

    assert launch["sft"]["num_processes"] == 1
    assert launch["grpo"]["num_processes"] == 2
    assert stage_summary(staged, "grpo") == launch["grpo"]


def test_sft_stage_runs_in_process_without_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(SMOKE)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.sft.train_sft",
        lambda **kwargs: calls.append(kwargs) or {"mode": "sft"},
    )
    runner = CommandRunner(cwd=tmp_path, dry_run=True)

    result = run_sft_stage(
        config=config,
        runner=runner,
        train_jsonl=tmp_path / "train.jsonl",
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "merged",
        run_name="in-process",
    )

    assert result == {"mode": "sft"}
    assert calls[0]["run_name"] == "in-process"
    assert runner.commands == []


def test_sft_stage_launches_worker_then_exports_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _with_profiles(load_config(SMOKE), sft=PROFILES / "ddp-2gpu.yaml")
    exports: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.checkpoint.export_merged_checkpoint",
        lambda **kwargs: exports.append(kwargs) or {"mode": "sft", "merged": True},
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.sft.train_sft",
        lambda **_: pytest.fail("worker path must not train in-process"),
    )
    runner = CommandRunner(cwd=tmp_path, dry_run=True)

    result = run_sft_stage(
        config=config,
        runner=runner,
        train_jsonl=tmp_path / "train.jsonl",
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "merged",
        run_name="worker",
    )

    assert result["merged"] is True
    (command,) = runner.commands
    assert command["name"] == "sft_worker"
    assert command["command"][1:5] == [
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(PROFILES / "ddp-2gpu.yaml"),
    ]
    assert command["command"][-8:] == [
        "--config",
        str(config.source),
        "--train-jsonl",
        str(tmp_path / "train.jsonl"),
        "--adapter-dir",
        str(tmp_path / "adapter"),
        "--run-name",
        "worker",
    ]
    assert exports == [
        {
            "base_model": config.model,
            "base_kwargs": {
                "trust_remote_code": True,
                "revision": config.model_revision,
            },
            "adapter_dir": tmp_path / "adapter",
            "output_dir": tmp_path / "merged",
        }
    ]


def test_grpo_stage_writes_task_ids_for_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(SMOKE)
    config = _with_profiles(
        replace(config, harness=replace(config.harness, concurrency=2)),
        grpo=PROFILES / "ddp-2gpu.yaml",
    )
    monkeypatch.setattr(
        "posttrainarena.benchflow_pipeline.checkpoint.export_merged_checkpoint",
        lambda **kwargs: {"mode": "grpo", **kwargs},
    )
    runner = CommandRunner(cwd=tmp_path, dry_run=True)
    jobs_dir = tmp_path / "jobs" / "grpo-train"

    result = run_grpo_stage(
        config=config,
        runner=runner,
        model=str(tmp_path / "sft-merged"),
        tasks_dir=tmp_path / "tasks",
        task_ids=["task-a", "task-b"],
        jobs_dir=jobs_dir,
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "merged",
        run_name="worker-grpo",
    )

    assert (jobs_dir / "task_ids.txt").read_text() == "task-a\ntask-b\n"
    assert result["base_model"] == str(tmp_path / "sft-merged")
    assert result["base_kwargs"] == {"trust_remote_code": True}
    (command,) = runner.commands
    assert command["name"] == "grpo_worker"
    assert "--task-ids-file" in command["command"]
    assert command["command"][command["command"].index("--task-ids-file") + 1] == str(
        jobs_dir / "task_ids.txt"
    )


def test_grpo_stage_rejects_a_profile_the_recipe_cannot_feed(tmp_path: Path) -> None:
    config = _with_profiles(load_config(SMOKE), grpo=PROFILES / "ddp-2gpu.yaml")

    with pytest.raises(ValueError, match="leaves no rollout slot"):
        run_grpo_stage(
            config=config,
            runner=CommandRunner(cwd=tmp_path, dry_run=True),
            model=config.model,
            tasks_dir=tmp_path,
            task_ids=["task-a"],
            jobs_dir=tmp_path / "jobs",
            adapter_dir=tmp_path / "adapter",
            output_dir=tmp_path / "merged",
            run_name="rejected",
        )


def test_worker_topology_must_match_the_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate
    import torch

    state = SimpleNamespace(num_processes=1, process_index=0, device="cpu")
    monkeypatch.setattr(accelerate, "PartialState", lambda **kwargs: state)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert verify_worker_topology(None) == {
        "process_index": 0,
        "num_processes": 1,
        "device": "cpu",
    }
    with pytest.raises(RuntimeError, match="expects 2"):
        verify_worker_topology(LaunchProfile.load(PROFILES / "ddp-2gpu.yaml"))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="sees several GPUs"):
        verify_worker_topology(LaunchProfile.load(PROFILES / "one-process.yaml"))


def test_worker_process_exits_nonzero_on_topology_mismatch(tmp_path: Path) -> None:
    recipe = tmp_path / "configs" / "ddp.toml"
    recipe.parent.mkdir()
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    recipe.write_text(
        SMOKE.read_text().replace(
            "[grpo]\n",
            f'[grpo]\naccelerate_config = "{PROFILES / "ddp-2gpu.yaml"}"\n',
        )
    )
    command = [
        sys.executable,
        "-m",
        "posttrainarena.benchflow_pipeline.cli",
        "grpo-worker",
        "--config",
        str(recipe),
        "--model",
        "unused",
        "--tasks-dir",
        str(tmp_path),
        "--task-ids-file",
        str(task_lists / "data-agent-train-15.txt"),
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--adapter-dir",
        str(tmp_path / "adapter"),
        "--run-name",
        "mismatch",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=300,
    )

    assert completed.returncode != 0
    assert "the launch profile expects 2" in completed.stderr
    assert not (tmp_path / "adapter").exists()
