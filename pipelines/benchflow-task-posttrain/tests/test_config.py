from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from posttrainarena.benchflow_pipeline.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_example_config_is_valid_and_pinned() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")

    assert config.model == "Qwen/Qwen3-4B"
    assert config.train_dataset.repo_id == "benchflow/data_agent_rl_environment_train"
    assert config.eval_dataset.repo_id == "benchflow/data_agent_rl_environment_eval"
    assert len(config.train_dataset.revision) == 40
    assert len(config.eval_dataset.revision) == 40
    assert config.harness.agent == "opencode"
    assert config.harness.usage_tracking == "required"
    assert config.harness.external_directory_allow == ("/home/user/input/**",)
    assert config.harness.deny_bash_patterns == ("*<<*",)
    assert config.evaluation.base_model_env == "BENCHFLOW_BASE_MODEL"
    assert config.evaluation.student_model_env == "BENCHFLOW_ADAPTER_MODEL"
    assert config.evaluation.base_url_env == "BENCHFLOW_PROVIDER_BASE_URL"
    assert config.evaluation.control_url_env == "BENCHFLOW_MODEL_BRIDGE_CONTROL_URL"
    assert config.evaluation.api_key_env == "BENCHFLOW_PROVIDER_API_KEY"
    assert config.teacher.model == "glm/glm-5.1"
    assert config.teacher.require_all_tasks is True
    assert config.teacher.min_reward == 1.0
    assert config.teacher.min_verified == 15
    assert config.runtime.num_generations == 2
    assert config.grpo.run_policy == "on_reward"
    assert config.grpo.rollout_attempts == 2
    assert config.grpo.vllm_server_base_url_env == "TRL_VLLM_SERVER_BASE_URL"
    assert config.sft.max_length == 40960


def test_qwen35_full_recipe_uses_all_tasks_and_one_epoch_lora() -> None:
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-full.toml")
    train_ids = config.train_dataset.task_list.read_text().splitlines()
    eval_ids = config.eval_dataset.task_list.read_text().splitlines()

    assert config.model == "Qwen/Qwen3.5-9B"
    assert config.model_revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert len(train_ids) == len(set(train_ids)) == 2238
    assert len(eval_ids) == len(set(eval_ids)) == 366
    assert set(train_ids).isdisjoint(eval_ids)
    assert hashlib.sha256(config.train_dataset.task_list.read_bytes()).hexdigest() == (
        "29892e57b957f1708fc5ed0bdfd08c4d2bab2249a4735d923da1369b6b606215"
    )
    assert hashlib.sha256(config.eval_dataset.task_list.read_bytes()).hexdigest() == (
        "99185677aa47ef406838387db37914b88a1816f5b2e7892cbdcd336ce708ab57"
    )
    assert config.harness.agent == "opencode"
    assert config.sandbox == "docker"
    assert config.teacher.model == "openrouter/qwen/qwen3.5-397b-a17b"
    assert config.teacher.source_model == "Qwen/Qwen3.5-397B-A17B"
    assert config.teacher.source_revision == (
        "8472618112abcbd45acbcdc58436aff4233c23f7"
    )
    assert config.teacher.require_all_tasks is True
    assert config.teacher.min_verified == 2238
    assert config.runtime.max_completion_length == 40960
    assert config.runtime.num_generations == 8
    assert config.sft.num_train_epochs == 1.0
    assert config.sft.max_steps is None
    assert config.sft.lora_r == 16
    assert config.sft.lora_alpha == 32
    assert config.grpo.run_policy == "always"
    assert config.grpo.num_train_epochs == 1.0
    assert config.grpo.max_steps is None
    assert config.grpo.lora_r == 16
    assert config.grpo.lora_alpha == 32
    assert config.grpo.log_completions is False
    assert config.grpo.gradient_accumulation_steps == 1
    assert config.grpo.generation_batch_size == 8
    assert config.grpo.require_reward_variance is True
    assert config.tracking.report_to == "none"
    assert config.evaluation.sync_base_to_vllm is True


def test_qwen35_soccer_canary_is_domain_matched_and_disjoint() -> None:
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-soccer-canary.toml")
    train_ids = config.train_dataset.task_list.read_text().splitlines()
    eval_ids = config.eval_dataset.task_list.read_text().splitlines()

    assert len(train_ids) == len(set(train_ids)) == 8
    assert len(eval_ids) == len(set(eval_ids)) == 3
    assert set(train_ids).isdisjoint(eval_ids)
    assert config.teacher.min_verified == 4
    assert config.teacher.require_all_tasks is False
    assert config.sft.num_train_epochs == 1.0
    assert config.sft.max_steps is None
    assert config.grpo.run_policy == "always"
    assert config.grpo.num_train_epochs == 1.0
    assert config.grpo.max_steps is None
    assert config.grpo.gradient_accumulation_steps == 1
    assert config.grpo.generation_batch_size == 2
    assert config.tracking.report_to == "none"
    assert config.evaluation.sync_base_to_vllm is True


def test_forced_grpo_smoke_config_bypasses_reward_gate() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-forced-grpo-smoke.toml")

    assert config.grpo.run_policy == "always"
    assert config.grpo.max_steps == 2


def test_skillsbench_e2e_smoke_is_disjoint_and_minimal() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-skillsbench-e2e-smoke.toml")

    assert config.train_dataset.repo_id == "benchflow/skillsbench"
    assert config.eval_dataset.repo_id == "benchflow/skillsbench"
    assert config.train_dataset.task_list.read_text().strip() == "3d-scan-calc"
    assert config.eval_dataset.task_list.read_text().strip() == "citation-check"
    assert config.sft.max_steps == 1
    assert config.grpo.max_steps == 1
    assert config.grpo.run_policy == "always"
    assert config.tracking.report_to == "none"


def test_config_accepts_always_grpo_run_policy(tmp_path: Path) -> None:
    source = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "always.toml"
    config_path.write_text(
        source.read_text().replace('run_policy = "on_reward"', 'run_policy = "always"')
    )

    config = load_config(config_path)

    assert config.grpo.run_policy == "always"


def test_config_rejects_unknown_grpo_run_policy() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, grpo=replace(config.grpo, run_policy="unconditional"))  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="grpo.run_policy must be on_reward or always",
    ):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rollout_attempts", 0, "grpo.rollout_attempts"),
        ("rollout_attempts", True, "grpo.rollout_attempts"),
        ("num_train_epochs", 0, "grpo.num_train_epochs"),
        ("max_steps", 0, "grpo.max_steps"),
        ("lora_r", 0, "grpo.lora_r"),
        ("lora_alpha", 0, "grpo.lora_alpha"),
        ("lora_dropout", 1.0, "grpo.lora_dropout"),
        ("log_completions", "yes", "grpo.log_completions"),
        ("generation_batch_size", 3, "grpo.generation_batch_size"),
        ("require_reward_variance", "yes", "grpo.require_reward_variance"),
        (
            "vllm_server_base_url_env",
            "",
            "grpo.vllm_server_base_url_env",
        ),
    ],
)
def test_config_rejects_invalid_opencode_grpo_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        grpo=replace(config.grpo, **{field: value}),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_rejects_non_opencode_harness() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(config, harness=replace(config.harness, agent="openhands"))

    with pytest.raises(ValueError, match="harness.agent must be opencode"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_model_env", "", "evaluation.base_model_env"),
        ("student_model_env", "", "evaluation.student_model_env"),
        ("base_url_env", "", "evaluation.base_url_env"),
        ("control_url_env", "", "evaluation.control_url_env"),
        ("api_key_env", "", "evaluation.api_key_env"),
    ],
)
def test_config_rejects_invalid_evaluation_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        evaluation=replace(config.evaluation, **{field: value}),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("skill_mode", "invalid", "harness.skill_mode"),
        ("usage_tracking", "auto", "harness.usage_tracking"),
        (
            "external_directory_allow",
            ("relative/path",),
            "harness.external_directory_allow",
        ),
        ("deny_bash_patterns", ("",), "harness.deny_bash_patterns"),
        ("concurrency", 0, "harness.concurrency"),
        ("concurrency", -1, "harness.concurrency"),
        ("concurrency", True, "harness.concurrency"),
        ("concurrency", "1", "harness.concurrency"),
        ("sandbox_setup_timeout_sec", 0, "harness.sandbox_setup_timeout_sec"),
        ("sandbox_setup_timeout_sec", True, "harness.sandbox_setup_timeout_sec"),
        ("agent_idle_timeout_sec", 0, "harness.agent_idle_timeout_sec"),
        ("agent_idle_timeout_sec", "300", "harness.agent_idle_timeout_sec"),
        ("agent_timeout_sec", 0, "harness.agent_timeout_sec"),
        ("agent_timeout_sec", True, "harness.agent_timeout_sec"),
        ("reasoning_effort", "", "harness.reasoning_effort"),
        ("reasoning_effort", 1, "harness.reasoning_effort"),
    ],
)
def test_config_rejects_invalid_harness_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        harness=replace(config.harness, **{field: value}),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "glm-5.1", "teacher.model"),
        ("model", "", "teacher.model"),
        ("model", 1, "teacher.model"),
        ("max_attempts", 0, "teacher max_attempts"),
        ("max_attempts", True, "teacher max_attempts"),
        ("min_verified", 0, "teacher max_attempts"),
        ("require_all_tasks", "yes", "teacher.require_all_tasks"),
        ("min_reward", -0.1, "teacher.min_reward"),
        ("min_reward", 1.1, "teacher.min_reward"),
        ("min_reward", True, "teacher.min_reward"),
        (
            "max_accepted_total_tokens",
            0,
            "teacher.max_accepted_total_tokens",
        ),
        (
            "max_accepted_total_tokens",
            True,
            "teacher.max_accepted_total_tokens",
        ),
        (
            "max_accepted_tool_calls",
            0,
            "teacher.max_accepted_tool_calls",
        ),
        (
            "max_accepted_tool_calls",
            "50",
            "teacher.max_accepted_tool_calls",
        ),
    ],
)
def test_config_rejects_invalid_teacher_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        teacher=replace(config.teacher, **{field: value}),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_rejects_mismatched_teacher_source_model() -> None:
    config = load_config(ROOT / "configs/qwen3.5-9b-data-agent-canary.toml")
    config = replace(
        config,
        teacher=replace(
            config.teacher,
            source_model="Qwen/Qwen3.5-35B-A3B",
        ),
    )

    with pytest.raises(ValueError, match="teacher.model must match"):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_train_epochs", 0, "sft.num_train_epochs"),
        ("max_steps", 0, "sft.max_steps"),
        ("lora_r", 0, "sft.lora_r"),
        ("lora_alpha", 0, "sft.lora_alpha"),
        ("lora_dropout", 1.0, "sft.lora_dropout"),
    ],
)
def test_config_rejects_invalid_sft_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        sft=replace(config.sft, **{field: value}),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_requires_harness_table(tmp_path: Path) -> None:
    source = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"
    text = source.read_text()
    harness_start = text.index("[harness]")
    teacher_start = text.index("[teacher]")
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "legacy.toml"
    config_path.write_text(text[:harness_start] + text[teacher_start:])

    with pytest.raises(ValueError, match=r"Missing required \[harness\] table"):
        load_config(config_path)


def test_config_rejects_non_table_harness(tmp_path: Path) -> None:
    source = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"
    text = source.read_text()
    harness_start = text.index("[harness]")
    teacher_start = text.index("[teacher]")
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "bad-harness.toml"
    config_path.write_text(
        "harness = 1\n" + text[:harness_start] + text[teacher_start:]
    )

    with pytest.raises(ValueError, match=r"\[harness\] must be a TOML table"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("external_directory_allow", '"/home/user/input/**"'),
        ("deny_bash_patterns", '"*<<*"'),
        ("deny_bash_patterns", "[]"),
        ("deny_bash_patterns", '["*<<*", 1]'),
    ],
)
def test_config_rejects_invalid_harness_array_shapes(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    source = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"
    text = source.read_text().replace(
        'usage_tracking = "required"',
        f'usage_tracking = "required"\n{field} = {value}',
    )
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "invalid-array.toml"
    config_path.write_text(text)

    with pytest.raises(ValueError, match=f"harness.{field}"):
        load_config(config_path)


def test_config_requires_evaluation_table(tmp_path: Path) -> None:
    source = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"
    text = source.read_text()
    evaluation_start = text.index("[evaluation]")
    teacher_start = text.index("[teacher]")
    task_lists = tmp_path / "task-lists"
    task_lists.mkdir()
    for name in ("data-agent-train-15.txt", "data-agent-eval-2.txt"):
        (task_lists / name).write_text((ROOT / "task-lists" / name).read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    config_path = configs / "missing-evaluation.toml"
    config_path.write_text(text[:evaluation_start] + text[teacher_start:])

    with pytest.raises(ValueError, match=r"Missing required \[evaluation\] table"):
        load_config(config_path)


def test_config_accepts_non_default_opencode_harness_values() -> None:
    config = load_config(ROOT / "configs/qwen3-4b-data-agent-smoke.toml")
    config = replace(
        config,
        harness=replace(
            config.harness,
            skill_mode="with-skill",
            concurrency=4,
            sandbox_setup_timeout_sec=600,
            agent_idle_timeout_sec=900,
            reasoning_effort="high",
        ),
    )

    config.validate()


def test_config_rejects_missing_task_list(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text(
        """
[model]
id = "model"
[train_dataset]
repo_id = "train"
revision = "abc"
task_list = "missing-train.txt"
[eval_dataset]
repo_id = "eval"
revision = "def"
task_list = "missing-eval.txt"
[harness]
agent = "opencode"
[evaluation]
"""
    )

    with pytest.raises(ValueError, match="does not exist"):
        load_config(config)
