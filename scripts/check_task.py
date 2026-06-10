#!/usr/bin/env python3
"""
Minimal task.md structural check — runs in CI without depending on the
benchflow CLI (which is in flux: the multi-level dogfood that knows the
task.md format lives on an unreleased upstream branch).

Validates:
- task.md exists and has YAML frontmatter (--- ... ---) + a Markdown body
- Required frontmatter fields are present
- environment/Dockerfile exists and starts with `FROM `
- verifier/{test.sh, test_outputs.py, verifier.md} all exist
- verifier/rubrics/ contains at least one *.md file
- oracle/solve.sh exists

This is the same shape `bench tasks check --level publication-grade`
enforces locally; remove this script and swap CI to the benchflow CLI
once upstream lands the new format on main.

Exit code: 0 if every task validates, 1 if any task has issues.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
REQUIRED_FRONTMATTER = (
    "version",
    "metadata",
    "agent",
    "verifier",
    "environment",
)
REQUIRED_METADATA = (
    "author_name",
    "author_email",
    "category",
)


def parse_yaml_keys(block: str) -> set[str]:
    """Cheap top-level key extractor — avoids importing PyYAML in CI."""
    keys: set[str] = set()
    for line in block.splitlines():
        if not line or line.startswith(("#", " ", "\t", "-")):
            continue
        if ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def parse_metadata_keys(block: str) -> set[str]:
    """Extract keys nested under `metadata:` (2-space indent expected)."""
    keys: set[str] = set()
    in_metadata = False
    for line in block.splitlines():
        stripped = line.lstrip()
        if line.startswith("metadata:"):
            in_metadata = True
            continue
        if in_metadata:
            if line and not line.startswith((" ", "\t")):
                in_metadata = False
                continue
            if stripped.startswith("#") or not stripped:
                continue
            if ":" in stripped:
                keys.add(stripped.split(":", 1)[0].strip())
    return keys


def check_task(task_dir: Path) -> list[str]:
    issues: list[str] = []

    # task.md ---------------------------------------------------------------
    task_md = task_dir / "task.md"
    if not task_md.exists():
        return [f"Missing required file: task.md"]

    text = task_md.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        issues.append("task.md must start with YAML frontmatter (--- ... ---)")
    else:
        frontmatter = m.group(1)
        body = text[m.end():]
        top_keys = parse_yaml_keys(frontmatter)
        for required in REQUIRED_FRONTMATTER:
            if required not in top_keys:
                issues.append(f"task.md frontmatter missing: {required}")
        metadata_keys = parse_metadata_keys(frontmatter)
        for required in REQUIRED_METADATA:
            if required not in metadata_keys:
                issues.append(f"task.md metadata.{required} required")
        if "## prompt" not in body:
            issues.append("task.md body must contain a '## prompt' section")

    # environment/ ----------------------------------------------------------
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        issues.append("Missing required file: environment/Dockerfile")
    else:
        # Skip comment lines and blanks; the first executable instruction
        # must be FROM.
        first_instr = next(
            (
                line
                for line in dockerfile.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ),
            "",
        )
        if not first_instr.upper().startswith("FROM "):
            issues.append("environment/Dockerfile first instruction must be FROM")

    # verifier/ -------------------------------------------------------------
    for f in ("verifier/test.sh", "verifier/test_outputs.py", "verifier/verifier.md"):
        if not (task_dir / f).exists():
            issues.append(f"Missing required file: {f}")
    rubrics = task_dir / "verifier" / "rubrics"
    if not rubrics.is_dir():
        issues.append("Missing required directory: verifier/rubrics/")
    else:
        if not any(p.suffix == ".md" for p in rubrics.iterdir()):
            issues.append("verifier/rubrics/ must contain at least one *.md rubric")

    # oracle/ ---------------------------------------------------------------
    if not (task_dir / "oracle" / "solve.sh").exists():
        issues.append("Missing required file: oracle/solve.sh")

    return issues


def iter_task_dirs(roots: Iterable[str]) -> Iterable[Path]:
    for root in roots:
        p = Path(root)
        if p.is_dir() and (p / "task.md").exists():
            yield p
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "task.md").exists():
                    yield child


def main(argv: list[str]) -> int:
    targets = argv[1:] or ["tasks"]
    overall_ok = True
    any_seen = False
    for task_dir in iter_task_dirs(targets):
        any_seen = True
        issues = check_task(task_dir)
        if issues:
            overall_ok = False
            print(f"✗ {task_dir.name} — {len(issues)} issue(s):")
            for i in issues:
                print(f"  → {i}")
        else:
            print(f"✓ {task_dir.name} — valid")
    if not any_seen:
        print("no task directories found")
        return 1
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
