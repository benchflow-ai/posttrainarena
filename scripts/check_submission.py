#!/usr/bin/env python3
"""
Team-submission structural check — runs in CI without external deps.

A competition entry is one directory under submissions/ owned by one
team, on one track:

    submissions/<team-entry>/
      submission.yaml        # flat key: value — team_name, contact_email, track
      envs/<env-name>/...    # track: environments — task packages
      skills/<skill-name>/   # track: skills — SKILL.md packages

Validates:
- submission.yaml exists with team_name, contact_email, and a known track
- the entry has at least one package for its track
- environment packages pass the same structural check as tasks/
  (delegated to scripts/check_task.py)
- skill packages contain a SKILL.md
- entry count is within the track's bounds: environments 50/100/200
  min/rec/max per entry, skills 20/50/100. Counts above max are errors;
  counts below min are warnings until the Phase 2 freeze (the published
  rules allow lowering the environments minimum to 25).

Directories whose name starts with "_" are skipped (scratch space).
Exit code: 0 if every entry validates (warnings allowed), 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_task import check_task  # noqa: E402

REQUIRED_FIELDS = ("team_name", "contact_email", "track")

# track -> (min, recommended, max) packages per entry
TRACK_BOUNDS = {
    "environments": (50, 100, 200),
    "skills": (20, 50, 100),
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

    lo, _rec, hi = TRACK_BOUNDS[track]
    if not packages:
        errors.append(f"entry declares track '{track}' but contains no packages under {root.name}/")
    elif len(packages) > hi:
        errors.append(f"{len(packages)} packages exceeds the {track} maximum of {hi} per entry")
    elif len(packages) < lo:
        warnings.append(
            f"{len(packages)} packages is below the {track} minimum of {lo} "
            "(warning until the Phase 2 freeze)"
        )

    return errors, warnings


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("submissions")
    if not root.is_dir():
        print(f"no {root}/ directory — nothing to check")
        return 0

    entries = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    if not entries:
        print("no team submissions yet")
        return 0

    overall_ok = True
    for entry in entries:
        errors, warnings = check_entry(entry)
        if errors:
            overall_ok = False
            print(f"✗ {entry.name} — {len(errors)} issue(s):")
            for e in errors:
                print(f"  → {e}")
        else:
            print(f"✓ {entry.name} — valid")
        for w in warnings:
            print(f"  ! {w}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
