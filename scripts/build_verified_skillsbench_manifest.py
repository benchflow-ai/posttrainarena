#!/usr/bin/env python3
"""Select one verified DeepSeek V4 Pro no-skills trace per SkillsBench task.

This is a pre-training safety gate. It never fabricates missing rows and exits
nonzero unless every active task has an eligible, content-unique trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def model_is_deepseek_v4_pro(value: object) -> bool:
    normalized = str(value or "").lower().replace("_", "-")
    return "deepseek-v4-pro" in normalized or "ds-v4-pro" in normalized


def reward(result: dict[str, Any]) -> float | None:
    rewards = result.get("rewards")
    value = rewards.get("reward") if isinstance(rewards, dict) else rewards
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def trajectory_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_invokes_skill(path: Path) -> bool:
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return True
        if not isinstance(event, dict):
            return True
        if event.get("type") != "tool_call":
            continue
        title = str(event.get("title") or "").lower()
        if title.startswith("invoke_skill:") or event.get("kind") == "skill":
            return True
    return False


def is_no_skills(result: dict[str, Any], result_path: Path) -> bool:
    mode = result.get("skill_mode")
    if mode is not None:
        return mode in {"no-skill", "no-skills", "without-skills"}
    # Some audited gap-fill artifacts predate the explicit skill_mode field.
    # Require both archive provenance and config-level absence of a skills dir.
    if "openhands-no-skills__" not in result_path.as_posix():
        return False
    config_path = result_path.parent / "config.json"
    if not config_path.is_file():
        return False
    config = load_json(config_path)
    if config.get("skills_dir") is not None:
        return False
    for scene in config.get("scenes") or []:
        if scene.get("skills_dir") is not None:
            return False
        for role in scene.get("roles") or []:
            if role.get("skills_dir") is not None:
                return False
    return True


def candidate(
    result_path: Path,
    active_tasks: set[str],
    require_llm_trajectory: bool,
) -> dict[str, Any] | None:
    result = load_json(result_path)
    task_id = result.get("task_name")
    if not isinstance(task_id, str) or task_id not in active_tasks:
        return None
    if reward(result) != 1.0:
        return None
    if not is_no_skills(result, result_path):
        return None
    if int(result.get("n_skill_invocations") or 0) != 0:
        return None
    if result.get("error") or result.get("verifier_error"):
        return None
    if not model_is_deepseek_v4_pro(result.get("model")):
        return None

    rollout_dir = result_path.parent
    acp_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    if not acp_path.is_file() or not acp_path.stat().st_size:
        return None
    if trajectory_invokes_skill(acp_path):
        return None
    llm_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    if require_llm_trajectory and (
        not llm_path.is_file() or not llm_path.stat().st_size
    ):
        return None
    return {
        "task_id": task_id,
        "reward": 1.0,
        "model": result.get("model"),
        "skill_mode": "no-skill",
        "n_skill_invocations": 0,
        "rollout_dir": str(rollout_dir.resolve()),
        "result_path": str(result_path.resolve()),
        "acp_trajectory_path": str(acp_path.resolve()),
        "llm_trajectory_path": str(llm_path.resolve()) if llm_path.is_file() else None,
        "trajectory_sha256": trajectory_hash(acp_path),
        "tool_calls": int(result.get("n_tool_calls") or 0),
        "total_tokens": (result.get("final_metrics") or {}).get("total_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        action="append",
        required=True,
        help="Artifact root to scan; repeat to merge archives and regenerated runs",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=87)
    parser.add_argument(
        "--expected-selected",
        type=int,
        default=None,
        help="Required verified training-task count; defaults to --expected-tasks",
    )
    parser.add_argument(
        "--require-llm-trajectory",
        action="store_true",
        help="Reject ACP-only rollouts that cannot be exported faithfully for SFT",
    )
    args = parser.parse_args()
    expected_selected = args.expected_selected or args.expected_tasks

    active_tasks = {
        path.name for path in args.tasks_dir.iterdir()
        if path.is_dir() and (path / "task.md").is_file()
    }
    if len(active_tasks) != args.expected_tasks:
        raise SystemExit(
            f"task roster has {len(active_tasks)} tasks; expected {args.expected_tasks}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    duplicate_artifacts = 0
    for artifacts_root in args.artifacts_root:
        for result_path in artifacts_root.rglob("result.json"):
            row = candidate(
                result_path,
                active_tasks,
                args.require_llm_trajectory,
            )
            if row is None:
                continue
            digest = row["trajectory_sha256"]
            if digest in seen_hashes:
                duplicate_artifacts += 1
                continue
            seen_hashes.add(digest)
            grouped[row["task_id"]].append(row)

    selected = []
    for task_id in sorted(active_tasks):
        rows = grouped.get(task_id, [])
        if not rows:
            continue
        rows.sort(
            key=lambda row: (
                row["llm_trajectory_path"] is not None,
                row["tool_calls"],
            ),
            reverse=True,
        )
        selected.append(rows[0])

    selected_hashes = [row["trajectory_sha256"] for row in selected]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise SystemExit("selected manifest contains duplicate trajectory content")

    selected_tasks = {row["task_id"] for row in selected}
    missing = sorted(active_tasks - selected_tasks)
    manifest = {
        "schema_version": 1,
        "policy": "one-verified-deepseek-v4-pro-no-skills-trace-per-active-task",
        "expected_task_count": args.expected_tasks,
        "selected_count": len(selected),
        "selected": selected,
        "missing_task_ids": missing,
        "eligible_unique_trace_count": sum(len(rows) for rows in grouped.values()),
        "duplicate_artifacts_ignored": duplicate_artifacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"selected {len(selected)}/{args.expected_tasks} tasks from "
        f"{manifest['eligible_unique_trace_count']} unique passing traces; "
        f"ignored {duplicate_artifacts} duplicate artifacts"
    )
    if len(selected) != expected_selected:
        print("missing:", ", ".join(missing))
        print(
            f"selected count {len(selected)} does not match required "
            f"{expected_selected}"
        )
        return 1
    print(f"validated manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
