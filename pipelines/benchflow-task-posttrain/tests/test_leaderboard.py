from __future__ import annotations

from posttrainarena.benchflow_pipeline.leaderboard import (
    merge_records,
    normalize_record,
    render_readme,
    render_space_app,
)


def _record(run_id: str, delta: float | None, status: str = "succeeded"):
    return {
        "run_id": run_id,
        "submission_id": f"submission-{run_id}",
        "status": status,
        "baseline_score": 0.1,
        "score_after_posttrain": None if delta is None else 0.1 + delta,
        "delta_score": delta,
    }


def test_leaderboard_upserts_and_ranks_successful_runs() -> None:
    records = merge_records(
        [],
        {**_record("run-a", None, "queued"), "job_id": "job-a"},
    )
    records = merge_records(records, _record("run-a", 0.1))
    records = merge_records(records, _record("run-b", 0.2))
    records = merge_records(records, _record("run-c", None, "running"))
    records = merge_records(records, {**_record("run-a", 0.3), "contact_email": "x@y"})

    assert [item["run_id"] for item in records] == ["run-a", "run-b", "run-c"]
    assert records[0]["job_id"] == "job-a"
    assert "contact_email" not in records[0]
    readme = render_readme(records)
    assert "| 1 | submission-run-a | succeeded |" in readme
    assert "| - | submission-run-c | running |" in readme


def test_leaderboard_rejects_non_numeric_scores() -> None:
    try:
        normalize_record(
            {
                "run_id": "run",
                "submission_id": "submission",
                "status": "succeeded",
                "delta_score": "bad",
            }
        )
    except ValueError as exc:
        assert "delta_score" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_space_app_is_pinned_to_leaderboard_dataset() -> None:
    source = render_space_app("benchflow/posttrainarena-leaderboard")
    assert "benchflow/posttrainarena-leaderboard" in source
    assert "hf_hub_download" in source
    compile(source, "app.py", "exec")
