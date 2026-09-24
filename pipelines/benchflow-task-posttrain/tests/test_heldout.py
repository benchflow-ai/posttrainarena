from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from posttrainarena.benchflow_pipeline.config import load_config
from posttrainarena.benchflow_pipeline.heldout import (
    arm_summary,
    cell_outcomes,
    paired_delta,
    score_report,
)
from posttrainarena.benchflow_pipeline.hf_jobs import create_job_bundle
from posttrainarena.benchflow_pipeline.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs/qwen3-4b-data-agent-smoke.toml"


def _outcomes(values: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    """Per-task cells: a reward, or None for an infrastructure error."""
    return {
        task_id: (
            {"reward": None, "passed": None, "infra_error": True}
            if reward is None
            else {"reward": reward, "passed": reward >= 1.0, "infra_error": False}
        )
        for task_id, reward in values.items()
    }


def _trials(table: dict[str, list[float | None]]) -> list[dict[str, dict[str, Any]]]:
    count = len(next(iter(table.values())))
    return [
        _outcomes({task_id: rewards[k] for task_id, rewards in table.items()})
        for k in range(count)
    ]


BASELINE_TB2 = {"a": [1.0, 0.0], "b": [0.0, 0.0], "c": [1.0, 1.0]}
FINAL_TB2 = {"a": [1.0, 1.0], "b": [1.0, 0.0], "c": [1.0, 1.0]}
BASELINE_LHTB = {"x": [0.0, 0.0], "y": [0.0, 1.0]}
FINAL_LHTB = {"x": [1.0, 0.0], "y": [1.0, 1.0]}


def test_paired_delta_uses_task_and_trial_variation() -> None:
    result = paired_delta(["a", "b", "c"], _trials(BASELINE_TB2), _trials(FINAL_TB2))

    # d = (0.5, 0.5, 0); s_d^2 = 1/12; SE = sqrt(s_d^2 / 3) = 1/6.
    assert result["paired_task_count"] == 3
    assert result["baseline_pass_rate"] == pytest.approx(0.5)
    assert result["final_pass_rate"] == pytest.approx(5 / 6)
    assert result["delta"] == pytest.approx(1 / 3)
    assert result["stderr"] == pytest.approx(1 / 6)
    assert result["ci95"][0] == pytest.approx(1 / 3 - 1.959963984540054 / 6)
    # Trial part: mean over tasks of (v_base/K + v_final/K) = (0.25 + 0.25 + 0) / 3.
    components = result["variance_components"]
    assert components["total"] == pytest.approx(1 / 12)
    assert components["trial"] == pytest.approx(1 / 6)
    assert components["task"] == 0.0
    assert result["stderr_fixed_tasks"] == pytest.approx(math.sqrt((1 / 6) / 3))


def test_arm_summary_excludes_infra_cells_instead_of_scoring_zero() -> None:
    trials = _trials({"a": [1.0, None], "b": [0.0, 0.0], "c": [None, None]})

    arm = arm_summary(["a", "b", "c"], trials)

    assert arm["scored_task_count"] == 2
    assert arm["unscored_task_ids"] == ["c"]
    assert arm["infra_error_cells"] == 3
    assert arm["cells"] == 6
    # a passes its only scored trial, b never passes: (1 + 0) / 2.
    assert arm["pass_rate"] == pytest.approx(0.5)
    assert arm["stderr"] == pytest.approx(math.sqrt(0.5 / 2))
    assert arm["trial_pass_rates"] == [pytest.approx(0.5), pytest.approx(0.0)]


def test_paired_delta_drops_tasks_missing_from_either_arm() -> None:
    baseline = _trials({"a": [0.0], "b": [None]})
    final = _trials({"a": [1.0], "b": [1.0]})

    result = paired_delta(["a", "b"], baseline, final)

    assert result["paired_task_count"] == 1
    assert result["excluded_task_ids"] == ["b"]
    assert result["delta"] == 1.0
    assert result["stderr"] is None
    assert result["variance_components"]["trial"] is None


def test_score_report_pools_suites_as_strata() -> None:
    report = score_report(
        suites=[
            {
                "name": "tb2",
                "dataset": {"repo_id": "r/tb2"},
                "task_ids": ["a", "b", "c"],
                "baseline": _trials(BASELINE_TB2),
                "final": _trials(FINAL_TB2),
            },
            {
                "name": "lhtb",
                "dataset": {"repo_id": "r/lhtb"},
                "task_ids": ["x", "y"],
                "baseline": _trials(BASELINE_LHTB),
                "final": _trials(FINAL_LHTB),
            },
        ],
        trials=2,
        baseline_stage="baseline_eval",
        final_stage="posttrain_eval",
    )

    assert report["schema_version"] == 2
    assert report["primary_suite"] == "tb2"
    assert [suite["name"] for suite in report["suites"]] == ["tb2", "lhtb"]
    lhtb = report["suites"][1]["delta"]
    assert lhtb["delta"] == pytest.approx(0.5)
    assert lhtb["stderr"] == 0.0
    pooled = report["pooled"]
    assert pooled["task_count"] == 5
    # (3 * 1/3 + 2 * 1/2) / 5, SE = sqrt((3/5)^2 (1/6)^2 + (2/5)^2 * 0).
    assert pooled["delta"]["delta"] == pytest.approx(0.4)
    assert pooled["delta"]["stderr"] == pytest.approx(0.1)
    assert pooled["delta"]["baseline_pass_rate"] == pytest.approx((3 * 0.5 + 2 * 0.25) / 5)
    assert pooled["baseline"]["pass_rate"] == pytest.approx(0.4)
    assert pooled["final"]["pass_rate"] == pytest.approx((3 * 5 / 6 + 2 * 0.75) / 5)


def test_score_report_without_final_arm_has_no_delta() -> None:
    report = score_report(
        suites=[
            {
                "name": "eval",
                "dataset": {},
                "task_ids": ["a", "b", "c"],
                "baseline": _trials(BASELINE_TB2),
                "final": None,
            }
        ],
        trials=2,
        baseline_stage="baseline_eval",
        final_stage=None,
    )

    assert report["suites"][0]["final"] is None
    assert report["suites"][0]["delta"] is None
    assert report["pooled"]["delta"] is None
    assert report["pooled"]["baseline"]["pass_rate"] == pytest.approx(0.5)


def test_cell_outcomes_reads_scored_rows_and_marks_infra_errors() -> None:
    payload = {
        "health": {
            "rows": [
                {
                    "task_id": "a",
                    "reward": None,
                    "scored": False,
                    "error": "sandbox create failed",
                    "verifier_error": None,
                    "valid_llm_trajectory": False,
                },
                {
                    "task_id": "a",
                    "reward": 1.0,
                    "scored": True,
                    "error": None,
                    "verifier_error": None,
                    "valid_llm_trajectory": True,
                },
                {
                    "task_id": "b",
                    "reward": 0.0,
                    "scored": True,
                    "error": "agent wall-clock budget exceeded",
                    "error_category": "timeout",
                    "verifier_error": None,
                    "valid_llm_trajectory": True,
                },
                {
                    "task_id": "c",
                    "reward": None,
                    "scored": False,
                    "error": "sandbox create failed",
                    "verifier_error": None,
                    "valid_llm_trajectory": False,
                },
            ]
        }
    }

    outcomes = cell_outcomes(payload, ["a", "b", "c"])

    assert outcomes["a"] == {"reward": 1.0, "passed": True, "infra_error": False}
    # A timeout is a scored failure, as in Terminal-Bench.
    assert outcomes["b"] == {"reward": 0.0, "passed": False, "infra_error": False}
    assert outcomes["c"]["infra_error"] is True
    assert cell_outcomes({"score": None}, ["a"]) is None


def _multi_suite_config(tmp_path: Path, *, trials: int = 2) -> Path:
    """The smoke config with two held-out suites (the smoke eval tasks split in two)."""
    text = SMOKE.read_text()
    eval_ids = (ROOT / "task-lists/data-agent-eval-2.txt").read_text().split()
    (tmp_path / "suite-a.txt").write_text(eval_ids[0] + "\n")
    (tmp_path / "suite-b.txt").write_text(eval_ids[1] + "\n")
    train_list = (ROOT / "task-lists/data-agent-train-15.txt").resolve()
    suites = (
        "[[eval_suites]]\n"
        'name = "tb2"\n'
        'repo_id = "benchflow/data_agent_rl_environment_eval"\n'
        'revision = "0ea976c79e3248c85737c4f7363484e4d47ce287"\n'
        'path = "tasks"\n'
        'task_list = "suite-a.txt"\n\n'
        "[[eval_suites]]\n"
        'name = "lhtb"\n'
        'repo_id = "benchflow/other_eval"\n'
        'revision = "1111111111111111111111111111111111111111"\n'
        'path = ""\n'
        'task_list = "suite-b.txt"\n\n'
    )
    text = re.sub(r"\[eval_dataset\]\n(?:[^\[\n][^\n]*\n)*\n", suites, text)
    text = text.replace(
        'task_list = "../task-lists/data-agent-train-15.txt"',
        f'task_list = "{train_list}"',
    )
    text = text.replace(
        'api_key_env = "BENCHFLOW_PROVIDER_API_KEY"\n',
        f'api_key_env = "BENCHFLOW_PROVIDER_API_KEY"\ntrials = {trials}\n',
    )
    text = text.replace('root = "../runs"', f'root = "{tmp_path / "runs"}"')
    path = tmp_path / "multi.toml"
    path.write_text(text)
    return path


def test_config_loads_eval_suites_and_trials(tmp_path: Path) -> None:
    config = load_config(_multi_suite_config(tmp_path, trials=3))

    assert [suite.name for suite in config.suites] == ["tb2", "lhtb"]
    assert config.eval_dataset == config.suites[0].dataset
    assert config.suites[1].dataset.path == ""
    assert config.suites[1].dataset.task_list == tmp_path / "suite-b.txt"
    assert config.evaluation.trials == 3


def test_legacy_config_has_one_implicit_suite() -> None:
    config = load_config(SMOKE)

    assert config.eval_suites == ()
    assert [suite.name for suite in config.suites] == ["eval"]
    assert config.suites[0].dataset == config.eval_dataset
    assert config.evaluation.trials == 1


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda text: text.replace("trials = 2", "trials = 0"), "evaluation.trials"),
        (lambda text: text.replace('name = "lhtb"', 'name = "tb2"'), "unique"),
        (lambda text: text.replace('name = "lhtb"', 'name = "LH TB"'), "names must match"),
        (
            lambda text: text + '\n[eval_dataset]\nrepo_id = "x"\nrevision = "y"\ntask_list = "suite-a.txt"\n',
            "not both",
        ),
        (lambda text: text.replace('task_list = "suite-b.txt"', 'task_list = "missing.txt"'), "eval_suites.lhtb.task_list"),
        (lambda text: text.replace('name = "lhtb"\n', ""), "missing: name"),
    ],
)
def test_config_rejects_invalid_eval_suites(tmp_path: Path, edit: Any, message: str) -> None:
    path = _multi_suite_config(tmp_path)
    path.write_text(edit(path.read_text()))

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_dry_run_evaluates_every_suite_and_trial(tmp_path: Path) -> None:
    config = load_config(_multi_suite_config(tmp_path, trials=2))
    pipeline = Pipeline(config, run_name="multi", dry_run=True)

    pipeline.run()
    saved = json.loads((pipeline.layout.reports / "score.json").read_text())
    names = [command["name"] for command in saved["commands"]]
    by_name = {command["name"]: command for command in saved["commands"]}

    assert "snapshot_eval_tasks" in names
    assert "snapshot_eval-lhtb_tasks" in names
    for stage in ("baseline_eval", "sft_eval", "posttrain_eval"):
        assert [name for name in names if name.startswith(stage)] == [
            stage,
            f"{stage}.tb2.t02",
            f"{stage}.lhtb.t01",
            f"{stage}.lhtb.t02",
        ]
    baseline = by_name["baseline_eval"]["command"]
    assert baseline[baseline.index("--jobs-dir") + 1] == str(pipeline.layout.jobs / "baseline")
    other = by_name["baseline_eval.lhtb.t02"]["command"]
    assert other[other.index("--jobs-dir") + 1] == str(
        pipeline.layout.jobs / "baseline-suites/lhtb/trial-02"
    )
    assert other[other.index("--tasks-dir") + 1] == str(
        pipeline.layout.root / "data/eval-lhtb"
    )
    # Legacy fields describe the primary suite; score_v2 lists every suite.
    assert saved["schema_version"] == 1
    assert saved["eval_task_ids"] == pipeline.suite_task_ids["tb2"]
    assert saved["eval_dataset"]["repo_id"] == "benchflow/data_agent_rl_environment_eval"
    assert [suite["name"] for suite in saved["score_v2"]["suites"]] == ["tb2", "lhtb"]
    assert saved["score_v2"]["trials"] == 2
    assert saved["score_v2"]["suites"][0]["baseline"] is None
    plan = json.loads((pipeline.layout.reports / "plan.json").read_text())
    assert [suite["name"] for suite in plan["eval_suites"]] == ["tb2", "lhtb"]


def test_heldout_scores_keep_legacy_fields_on_the_primary_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_multi_suite_config(tmp_path, trials=2))
    pipeline = Pipeline(config, run_name="scored", dry_run=False)
    tb2_id = pipeline.suite_task_ids["tb2"][0]
    lhtb_id = pipeline.suite_task_ids["lhtb"][0]
    rewards = {
        "baseline_eval": 0.0,
        "baseline_eval.tb2.t02": 0.0,
        "baseline_eval.lhtb.t01": 0.0,
        "baseline_eval.lhtb.t02": 1.0,
        "posttrain_eval": 1.0,
        "posttrain_eval.tb2.t02": 0.0,
        "posttrain_eval.lhtb.t01": 1.0,
        "posttrain_eval.lhtb.t02": None,  # infrastructure error: excluded, not scored 0
    }
    seen: list[tuple[str, str]] = []

    def fake_payload(**kwargs: Any) -> dict[str, Any]:
        stage = kwargs["stage"]
        seen.append((stage, str(kwargs["jobs_dir"])))
        reward = rewards[stage]
        task_id = kwargs["task_ids"][0]
        row = {
            "task_id": task_id,
            "reward": reward,
            "scored": reward is not None,
            "error": None if reward is not None else "sandbox create failed",
            "verifier_error": None,
            "valid_llm_trajectory": reward is not None,
        }
        return {"score": reward or 0.0, "health": {"rows": [row]}}

    monkeypatch.setattr(pipeline, "_evaluate_payload", fake_payload)
    baseline = pipeline._evaluate_heldout(
        stage="baseline_eval", jobs_name="baseline", model=config.model, policy_sha256="p"
    )
    final = pipeline._evaluate_heldout(
        stage="posttrain_eval", jobs_name="posttrain", model="/ckpt", policy_sha256="q"
    )
    summary = pipeline._write_score(
        baseline_score=baseline["primary_score"],
        sft_score=None,
        grpo_gate_score=None,
        final_score=final["primary_score"],
        final_model="/ckpt",
        grpo_planned=True,
        grpo_ran=True,
        baseline_heldout=baseline,
        final_heldout=final,
    )

    assert seen[0] == ("baseline_eval", str(pipeline.layout.jobs / "baseline"))
    assert seen[4] == ("posttrain_eval", str(pipeline.layout.jobs / "posttrain"))
    assert summary["baseline_score"] == 0.0
    assert summary["score_after_posttrain"] == 1.0
    assert summary["delta_score"] == 1.0
    v2 = summary["score_v2"]
    tb2, lhtb = v2["suites"]
    assert tb2["baseline"]["pass_rate"] == 0.0
    assert tb2["final"]["pass_rate"] == 0.5
    assert tb2["delta"]["delta"] == 0.5
    assert lhtb["final"]["infra_error_cells"] == 1
    assert lhtb["final"]["pass_rate"] == 1.0
    assert lhtb["delta"]["delta"] == 0.5
    assert v2["pooled"]["delta"]["delta"] == 0.5
    assert v2["final_stage"] == "posttrain_eval"
    outcomes = json.loads((pipeline.layout.reports / "eval_task_outcomes.json").read_text())
    final_cells = outcomes["arms"]["final"]["cells"]
    assert [cell["stage"] for cell in final_cells][-1] == "posttrain_eval.lhtb.t02"
    assert final_cells[-1]["outcomes"][lhtb_id]["infra_error"] is True
    assert final_cells[0]["outcomes"][tb2_id]["passed"] is True
    markdown = (pipeline.layout.reports / "SCORE.md").read_text()
    assert "| pooled | 2 |" in markdown


def test_job_bundle_copies_every_eval_suite_task_list(tmp_path: Path) -> None:
    bundle = create_job_bundle(
        config_path=_multi_suite_config(tmp_path),
        output_dir=tmp_path / "bundle",
        run_id="run-1",
        submission_id="team",
        team_name="Team",
        artifact_repo="benchflow/results",
        leaderboard_repo="benchflow/leaderboard",
        model_repo=None,
    )

    config = tomllib.loads(bundle.config_path.read_text())
    assert "eval_dataset" not in config
    assert [suite["task_list"] for suite in config["eval_suites"]] == [
        "task-lists/eval_suite-tb2.txt",
        "task-lists/eval_suite-lhtb.txt",
    ]
    assert (bundle.root / "task-lists/eval_suite-lhtb.txt").is_file()
