from __future__ import annotations

from types import SimpleNamespace

import pytest

from posttrainarena.benchflow_pipeline.distributed import (
    close_process_group,
    declare_fsdp_layer_classes,
    gather_peak_gpu_memory_mib,
    gather_records,
    is_main_process,
    num_processes,
    process_index,
    rollout_quota,
    subprocess_environment,
    wait_for_everyone,
)


def test_helpers_treat_a_trainer_without_accelerator_as_one_process() -> None:
    trainer = SimpleNamespace()

    assert process_index(trainer) == 0
    assert num_processes(trainer) == 1
    assert is_main_process(trainer) is True
    wait_for_everyone(trainer)
    assert gather_records(trainer, [{"rank": 0}]) == [{"rank": 0}]


def test_helpers_read_the_trainer_accelerator() -> None:
    barriers: list[str] = []
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            process_index=1,
            num_processes=2,
            is_main_process=False,
            wait_for_everyone=lambda: barriers.append("barrier"),
        )
    )

    assert process_index(trainer) == 1
    assert num_processes(trainer) == 2
    assert is_main_process(trainer) is False
    wait_for_everyone(trainer)
    assert barriers == ["barrier"]


def test_gather_records_uses_accelerate_collective_beyond_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate.utils

    monkeypatch.setattr(
        accelerate.utils,
        "gather_object",
        lambda values: [*values, {"rank": 1}],
    )
    trainer = SimpleNamespace(accelerator=SimpleNamespace(num_processes=2))

    assert gather_records(trainer, [{"rank": 0}]) == [{"rank": 0}, {"rank": 1}]


@pytest.mark.parametrize(
    ("total", "world_size", "expected"),
    [
        (8, 1, [8]),
        (8, 2, [4, 4]),
        (5, 2, [3, 2]),
        (2, 3, [1, 1, 0]),
    ],
)
def test_rollout_quota_splits_the_budget_lower_ranks_first(
    total: int, world_size: int, expected: list[int]
) -> None:
    assert [rollout_quota(total, world_size, rank) for rank in range(world_size)] == expected
    assert sum(expected) == total


def test_rollout_quota_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError):
        rollout_quota(0, 1, 0)
    with pytest.raises(ValueError):
        rollout_quota(4, 2, 2)


def test_subprocess_environment_drops_the_process_group_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc")
    monkeypatch.setenv("HF_HOME", "/cache")

    environment = subprocess_environment()

    assert environment["HF_HOME"] == "/cache"
    assert not {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR"} & set(environment)
    assert not any(key.startswith("TORCHELASTIC_") for key in environment)


def test_close_process_group_is_a_no_op_outside_a_group() -> None:
    close_process_group()


def test_gather_peak_gpu_memory_reports_every_rank_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import accelerate.utils

    monkeypatch.setattr(
        accelerate.utils,
        "gather_object",
        lambda values: [*values, {"peak_mib": 7}],
    )
    trainer = SimpleNamespace(accelerator=SimpleNamespace(num_processes=2))

    peaks = gather_peak_gpu_memory_mib(trainer)

    assert len(peaks) == 2
    assert peaks[1] == 7
    assert all(isinstance(value, int) and value >= 0 for value in peaks)


def test_declare_fsdp_layer_classes_names_only_present_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class DecoderLayer(torch.nn.Module):
        pass

    class Model(torch.nn.Module):
        _no_split_modules = ["DecoderLayer", "VisionBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.layer = DecoderLayer()

    monkeypatch.delenv("FSDP_TRANSFORMER_CLS_TO_WRAP", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    assert declare_fsdp_layer_classes(Model()) is None
    assert "FSDP_TRANSFORMER_CLS_TO_WRAP" not in __import__("os").environ

    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    assert declare_fsdp_layer_classes(Model()) == ["DecoderLayer"]
    assert __import__("os").environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] == "DecoderLayer"

    monkeypatch.setenv("FSDP_TRANSFORMER_CLS_TO_WRAP", "Custom")
    assert declare_fsdp_layer_classes(Model()) is None
