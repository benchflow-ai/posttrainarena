"""Atomic Hub-dataset leaderboard records for PostTrain Arena runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    required = ("run_id", "submission_id", "status")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"leaderboard record missing: {', '.join(missing)}")
    clean = dict(record)
    clean.setdefault("schema_version", 1)
    clean.setdefault("updated_at", utc_now())
    for key in (
        "baseline_score",
        "score_after_posttrain",
        "delta_score",
    ):
        value = clean.get(key)
        if value is not None and (
            not isinstance(value, int | float) or isinstance(value, bool)
        ):
            raise ValueError(f"{key} must be numeric or null")
    clean.pop("contact_email", None)
    return clean


def merge_records(
    records: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    normalized = normalize_record(record)
    by_run = {str(item["run_id"]): dict(item) for item in records}
    existing = by_run.get(str(normalized["run_id"]), {})
    existing.update(
        {key: value for key, value in normalized.items() if value is not None}
    )
    by_run[str(normalized["run_id"])] = existing
    return sorted(
        by_run.values(),
        key=lambda item: (
            item.get("status") != "succeeded",
            -(float(item.get("delta_score") or 0.0)),
            str(item.get("run_id")),
        ),
    )


def render_readme(records: list[dict[str, Any]]) -> str:
    lines = [
        "---",
        "pretty_name: PostTrain Arena Leaderboard",
        "license: other",
        "---",
        "",
        "# PostTrain Arena Leaderboard",
        "",
        "Continuously updated results from pinned PostTrain Arena jobs.",
        "",
        "| Rank | Submission | Status | Baseline | Final | Delta | Run |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    rank = 0
    for record in records:
        if record.get("status") == "succeeded":
            rank += 1
            rank_text = str(rank)
        else:
            rank_text = "-"
        artifact_url = str(record.get("artifact_url") or "")
        run_text = (
            f"[{record['run_id']}]({artifact_url})"
            if artifact_url
            else str(record["run_id"])
        )
        lines.append(
            "| {rank} | {submission} | {status} | {baseline} | {final} | "
            "{delta} | {run} |".format(
                rank=rank_text,
                submission=record.get("submission_id", ""),
                status=record.get("status", ""),
                baseline=_score(record.get("baseline_score")),
                final=_score(record.get("score_after_posttrain")),
                delta=_score(record.get("delta_score")),
                run=run_text,
            )
        )
    return "\n".join(lines) + "\n"


def _score(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def publish_record(
    *,
    repo_id: str,
    record: dict[str, Any],
    token: str | None = None,
    private: bool = False,
    max_attempts: int = 4,
) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        info = api.repo_info(repo_id, repo_type="dataset")
        try:
            path = hf_hub_download(
                repo_id,
                "leaderboard.json",
                repo_type="dataset",
                revision=info.sha,
                token=token,
                force_download=True,
            )
            current = json.loads(open(path, encoding="utf-8").read())
        except EntryNotFoundError:
            current = []
        records = merge_records(current, record)
        payload = json.dumps(records, indent=2, sort_keys=True) + "\n"
        run_payload = json.dumps(
            normalize_record(record), indent=2, sort_keys=True
        ) + "\n"
        operations = [
            CommitOperationAdd(
                path_in_repo="leaderboard.json",
                path_or_fileobj=payload.encode(),
            ),
            CommitOperationAdd(
                path_in_repo=f"runs/{record['run_id']}.json",
                path_or_fileobj=run_payload.encode(),
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=render_readme(records).encode(),
            ),
        ]
        try:
            commit = api.create_commit(
                repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Update run {record['run_id']}",
                parent_commit=info.sha,
            )
            return {
                "repo_id": repo_id,
                "commit": commit.oid,
                "url": commit.commit_url,
                "record_count": len(records),
            }
        except HfHubHTTPError as exc:
            last_error = exc
            if exc.response is None or exc.response.status_code not in {409, 412}:
                raise
    raise RuntimeError("leaderboard update conflicted repeatedly") from last_error


def render_space_app(leaderboard_repo: str) -> str:
    return f'''"""Live PostTrain Arena leaderboard backed by a Hub dataset."""

import json

import gradio as gr
from huggingface_hub import hf_hub_download


LEADERBOARD_REPO = {leaderboard_repo!r}


def load_rows():
    path = hf_hub_download(
        LEADERBOARD_REPO,
        "leaderboard.json",
        repo_type="dataset",
        force_download=True,
    )
    records = json.loads(open(path, encoding="utf-8").read())
    rows = []
    rank = 0
    for record in records:
        if record.get("status") == "succeeded":
            rank += 1
            shown_rank = rank
        else:
            shown_rank = None
        rows.append(
            [
                shown_rank,
                record.get("submission_id"),
                record.get("status"),
                record.get("baseline_score"),
                record.get("score_after_posttrain"),
                record.get("delta_score"),
                record.get("run_id"),
                record.get("artifact_url"),
            ]
        )
    return rows


with gr.Blocks(title="PostTrain Arena Leaderboard") as demo:
    gr.Markdown("# PostTrain Arena Leaderboard")
    gr.Markdown(
        "Continuously updated results from pinned Hugging Face Jobs. "
        "Scores link to immutable run artifacts."
    )
    table = gr.Dataframe(
        headers=[
            "Rank",
            "Submission",
            "Status",
            "Baseline",
            "Final",
            "Delta",
            "Run",
            "Artifacts",
        ],
        value=load_rows,
        interactive=False,
    )
    refresh = gr.Button("Refresh")
    refresh.click(load_rows, outputs=table)
    timer = gr.Timer(60)
    timer.tick(load_rows, outputs=table)


if __name__ == "__main__":
    demo.launch()
'''


def deploy_space(
    *,
    space_repo: str,
    leaderboard_repo: str,
    token: str | None = None,
    private: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(
        space_repo,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )
    readme = (
        "---\n"
        "title: PostTrain Arena Leaderboard\n"
        "emoji: 🏟️\n"
        "colorFrom: blue\n"
        "colorTo: green\n"
        "sdk: gradio\n"
        "app_file: app.py\n"
        "pinned: false\n"
        "---\n\n"
        f"Live view of `{leaderboard_repo}`.\n"
    )
    operations = [
        CommitOperationAdd(
            path_in_repo="app.py",
            path_or_fileobj=render_space_app(leaderboard_repo).encode(),
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme.encode(),
        ),
        CommitOperationAdd(
            path_in_repo="requirements.txt",
            path_or_fileobj=(
                "gradio>=5,<7\nhuggingface_hub>=0.36,<2\n"
            ).encode(),
        ),
    ]
    commit = api.create_commit(
        space_repo,
        repo_type="space",
        operations=operations,
        commit_message="Deploy PostTrain Arena leaderboard",
    )
    return {
        "space_repo": space_repo,
        "leaderboard_repo": leaderboard_repo,
        "commit": commit.oid,
        "url": f"https://huggingface.co/spaces/{space_repo}",
    }
