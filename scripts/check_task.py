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

It checks structure only: that the files, keys and prompt section exist,
not their contents. The untouched template passes, placeholders and all.
scripts/run_local.sh replays the oracle and an empty trial, and the Arena's
validate checks the metadata values (category, license, origin).

One content check warns without failing: a verifier/test.sh that downloads
when it runs (apt-get, curl, pip or uv installs, uvx) while task.md sets
allow_internet: false. In a sandbox without network, which is how
scripts/run_local.sh and the Arena's oracle and no-op controls run a task,
the download fails or hangs, so even the reference solution can't pass. A test.sh that
runs pytest from the image when the image has it (command -v pytest) and a
Dockerfile that installs pytest pass, as the template does.

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

# What verifier/test.sh downloads with when it runs: package installs, curl or wget, uvx and the like.
DOWNLOADS = re.compile(
    r"\b(?P<install>(?:apt-get|apt|apk|yum|dnf|pip3?|uv|conda|mamba|npm|pnpm|yarn|gem|cargo|go)\s+(?:-\S+\s+)*"
    r"(?:install|add|get|sync|tool))\b"
    r"|\b(?P<fetch>curl|wget|uvx|pipx\s+run|npx|git\s+clone)\b"
)
# A test.sh that runs pytest from the image when the image has it (the template's `command -v "$PYTEST_BIN"`).
PYTEST_IN_IMAGE = re.compile(r"\b(?:command\s+-v|which|type\s+-[pP]|hash)\s+[\"']?(?:\$\{?PYTEST_BIN\}?|pytest)(?![\w-])")
OFFLINE = {"false", "no", "off", "0"}


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


def block_values(frontmatter: str, block: str) -> dict[str, str]:
    """Scalar keys directly under a top-level block (`environment:`, `sandbox:`), comments dropped."""
    values: dict[str, str] = {}
    inside, indent = False, None
    for line in frontmatter.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            key, _, rest = line.partition(":")
            inside, indent = key.strip() == block and not rest.strip(), None
            continue
        if not inside or ":" not in line:
            continue
        depth = len(line) - len(line.lstrip())
        indent = depth if indent is None else indent
        if depth == indent:
            key, _, raw = line.strip().partition(":")
            values[key.strip()] = re.split(r"\s+#", raw, maxsplit=1)[0].strip().strip("\"'")
    return values


def offline(frontmatter: str) -> bool:
    """True when task.md turns the sandbox's network off: allow_internet: false under environment: (or sandbox:, in
    BenchFlow's own format), or BenchFlow's network_mode: no-network."""
    for block in ("environment", "sandbox"):
        values = block_values(frontmatter, block)
        if values.get("allow_internet", "").lower() in OFFLINE or values.get("network_mode", "").lower() == "no-network":
            return True
    return False


def image_installs_pytest(dockerfile: str) -> bool:
    """A RUN instruction of the Dockerfile installs pytest (pip, uv, apt's python3-pytest ...)."""
    joined = re.sub(r"\\\n", " ", dockerfile)
    return any(
        re.match(r"\s*RUN\b", line, re.I) and re.search(r"\binstall\b", line) and re.search(r"(?<![\w.])pytest(?!\w)", line)
        for line in joined.splitlines()
    )


def warnings_for(task_dir: Path) -> list[str]:
    """Content problems that don't fail the check: a verifier that needs the network the task turns off."""
    task_md, test_sh = task_dir / "task.md", task_dir / "verifier" / "test.sh"
    if not task_md.is_file() or not test_sh.is_file():
        return []
    m = FRONTMATTER_RE.match(task_md.read_text(encoding="utf-8", errors="replace"))
    if not m or not offline(m.group(1)):
        return []
    code = [line for line in test_sh.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    tools = []
    for line in code:
        for hit in DOWNLOADS.finditer(line):
            words = (hit.group("install") or hit.group("fetch")).split()
            tools.append(f"{words[0]} {words[-1]}" if hit.group("install") else " ".join(words))
    tools = list(dict.fromkeys(tools))
    if not tools:
        return []
    dockerfile = task_dir / "environment" / "Dockerfile"
    has_pytest = dockerfile.is_file() and image_installs_pytest(dockerfile.read_text(encoding="utf-8", errors="replace"))
    fallback = any(PYTEST_IN_IMAGE.search(line) for line in code)
    if fallback and has_pytest:
        return []   # the download is a fallback for an image without pytest; this image has it
    why = (" (it runs pytest from the image when the image has it, but environment/Dockerfile doesn't install pytest)"
           if fallback else "")
    return [
        f"verifier/test.sh downloads when it runs ({', '.join(tools)}){why}, but task.md sets allow_internet: false. "
        "In a sandbox without network (scripts/run_local.sh, the Arena's oracle and no-op controls) the download fails "
        "or hangs, so even the reference solution can't pass. Install what the verifier needs in environment/Dockerfile and run it "
        "from the image, as starting-kit/template does (pytest==8.4.1 and pytest-json-ctrf==0.3.5)."
    ]


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
    targets = argv[1:] or ["starting-kit/examples", "starting-kit/template"]
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
            print(f"✓ {task_dir.name} — structure valid")
        for w in warnings_for(task_dir):
            print(f"  ⚠ warning: {w}")
    if not any_seen:
        print("no task directories found")
        return 1
    print(
        "This checks structure only: that the required files, frontmatter keys and "
        "'## prompt' section exist, not their contents (the template's placeholders pass) "
        "or whether the verifier and oracle work. Its one content check, a verifier that "
        "downloads while the task turns the network off, is a warning."
    )
    if overall_ok:
        print(
            "Next: scripts/run_local.sh <task> (the oracle must score 1) and "
            "scripts/run_local.sh <task> --skip-oracle (an empty trial must not)."
        )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
