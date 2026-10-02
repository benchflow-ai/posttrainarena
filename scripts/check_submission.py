#!/usr/bin/env python3
"""
Team-submission structural check — runs in CI without external deps.

A competition entry is one directory under submissions/ owned by one
team, on one track:

    submissions/<team-entry>/
      submission.yaml        # flat key: value — team_name, track
      envs/<env-name>/...    # track: environments — task packages
      skills/<skill-name>/   # track: skills — SKILL.md packages

Validates:
- submission.yaml exists with team_name and a known track
- the entry has at least one package for its track
- environment packages pass the same structural check as the
  starting-kit examples (delegated to scripts/check_task.py)
- skill packages contain a SKILL.md
- entry count is within the track's bounds: environments 1–200 per entry,
  the hosted Arena's range; legacy skills 20–100. Counts above the maximum
  are errors; a skills entry below 20 gets a warning.

Usage:
    python3 scripts/check_submission.py                  # every entry under submissions/
    python3 scripts/check_submission.py my-collection    # one collection: the folder that holds submission.yaml
    python3 scripts/check_submission.py some/folder      # every collection in a folder of collections

Directories whose name starts with "_" are skipped (scratch space).
Exit code: 0 if every entry validates (warnings allowed), 1 otherwise, 2 for a
usage error such as a folder that does not exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_task import check_task  # noqa: E402

REQUIRED_FIELDS = ("team_name", "track")

# track -> (min, max) packages per entry. Environments: the hosted Arena accepts
# 1-200 task packages per collection. Skills: the legacy format's bounds.
TRACK_BOUNDS = {
    "environments": (1, 200),
    "skills": (20, 100),
}


def parse_flat_yaml(text: str) -> dict[str, str]:
    """Top-level `key: value` lines only — no nesting, no PyYAML."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith(("#", " ", "\t", "-")):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.split("#", 1)[0].strip().strip("\"'")
    return out


def check_entry(entry: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = entry / "submission.yaml"
    if not manifest_path.exists():
        return ["Missing required file: submission.yaml"], []

    manifest = parse_flat_yaml(manifest_path.read_text(encoding="utf-8"))
    for field in REQUIRED_FIELDS:
        if not manifest.get(field):
            errors.append(f"submission.yaml missing: {field}")

    track = manifest.get("track", "")
    if track and track not in TRACK_BOUNDS:
        errors.append(
            f"submission.yaml track must be one of {sorted(TRACK_BOUNDS)}, got: {track!r}"
        )
        return errors, warnings
    if not track:
        return errors, warnings

    if track == "environments":
        root = entry / "envs"
        packages = sorted(
            p for p in root.iterdir() if p.is_dir() and (p / "task.md").exists()
        ) if root.is_dir() else []
        for pkg in packages:
            for issue in check_task(pkg):
                errors.append(f"envs/{pkg.name}: {issue}")
    else:
        root = entry / "skills"
        packages = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        for pkg in packages:
            if not (pkg / "SKILL.md").exists():
                errors.append(f"skills/{pkg.name}: missing SKILL.md")

    lo, hi = TRACK_BOUNDS[track]
    if not packages:
        errors.append(f"entry declares track '{track}' but contains no packages under {root.name}/")
    elif len(packages) > hi:
        errors.append(f"{len(packages)} packages exceeds the {track} maximum of {hi} per entry")
    elif len(packages) < lo:
        warnings.append(
            f"{len(packages)} packages is below the legacy {track} track's minimum of {lo} "
            "(a warning, not an error)"
        )

    return errors, warnings


def collections_in(folder: Path) -> tuple[list[Path], str | None]:
    """The collections a folder argument names: itself when it holds submission.yaml, else its subfolders (a folder of
    collections, like submissions/). The second value explains a folder that is a task package or an envs/ folder."""
    if (folder / "submission.yaml").is_file():
        return [folder], None
    if (folder / "task.md").is_file():
        return [], (f"{folder} is a task package, not a collection. Check it with scripts/check_task.py {folder}; "
                    "a collection is the folder that holds submission.yaml and envs/.")
    children = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if any((c / "task.md").is_file() for c in children) and not any((c / "submission.yaml").is_file() for c in children):
        return [], (f"{folder} holds task packages, not collections. Pass the collection folder, the one that holds "
                    f"submission.yaml and envs/, or check the tasks with scripts/check_task.py {folder}")
    return children, None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="check_submission.py",
        description="Check team collections' structure: submission.yaml (team_name, track), then every "
                    "package under envs/ (the same check as scripts/check_task.py) and the package count (environments 1-200).",
        epilog="This checks structure only, not metadata values or whether a task works: scripts/run_local.sh replays the "
               "oracle and an empty trial, and the Arena's validate checks category, license and origin.")
    parser.add_argument("folders", nargs="*", metavar="FOLDER",
                        help="a collection (the folder that holds submission.yaml and envs/) or a folder of collections; "
                             "default: submissions/")
    args = parser.parse_args(argv[1:])
    for folder in args.folders:
        if not Path(folder).is_dir():
            parser.error(f"{folder}: no such folder")
    if not args.folders:
        root = Path("submissions")
        if not root.is_dir():
            print(f"no {root}/ directory — nothing to check")
            return 0
        entries, why = collections_in(root)
        if why:
            print(f"✗ {why}")
            return 1
        if not entries:
            print("no team submissions yet")
            return 0
    else:
        entries = []
        for folder in args.folders:
            found, why = collections_in(Path(folder))
            if why:
                print(f"✗ {why}")
                return 1
            if not found:
                print(f"✗ {folder} holds no collection: a collection is a folder with submission.yaml and envs/.")
                return 1
            entries += found

    overall_ok = True
    for entry in entries:
        errors, warnings = check_entry(entry)
        if errors:
            overall_ok = False
            print(f"✗ {entry.name} — {len(errors)} issue(s):")
            for e in errors:
                print(f"  → {e}")
        else:
            print(f"✓ {entry.name} — structure valid")
        for w in warnings:
            print(f"  ! {w}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
