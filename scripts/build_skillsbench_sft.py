#!/usr/bin/env python3
"""Build an SFT corpus from SkillsBench oracle solutions, holding out the eval set.

Why this corpus exists
----------------------
The first end-to-end reading was a significant *negative*: dense Qwen3.6-27B
trained on 200 TMax rows scored -0.243 on SkillsBench (CI [-0.493, -0.021]),
even though the memorization probe confirms the checkpoint learned its training
data. TMax is terminal-agentic environment rollouts; SkillsBench is
skills-shaped tasks. The tune worked and transferred negatively.

The honest fix is in-domain data. Every SkillsBench task ships an `oracle/`
directory containing a working solution, so the benchmark carries its own
training signal. This renders (task instruction -> oracle solution) pairs.

The held-out set is the whole point. Tasks scored by the hillclimb spec are
excluded here, so a positive delta means the model generalized to tasks it
never saw rather than memorizing the answers it was graded on. Training on the
eval tasks would produce a large, meaningless number.

    python3 scripts/build_skillsbench_sft.py \\
        --tasks-dir <skillsbench>/tasks --out sft_skillsbench.jsonl \\
        --exclude dialogue-parser citation-check ...
"""

from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path

# Solution files worth learning from. Shell drives the task; the others carry
# the actual reasoning. Anything else (data snapshots, fixtures) is noise.
CODE_SUFFIXES = {".sh", ".py", ".js", ".ts", ".sql", ".r", ".jl"}
LANG = {".sh": "bash", ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".sql": "sql", ".r": "r", ".jl": "julia"}

SYSTEM = (
    "You are an expert software engineer working in a Linux terminal. You are given a "
    "task with a workspace at /app. Produce a complete, working solution: explain your "
    "approach briefly, then give the exact files and commands needed."
)


def instruction(task_md: Path) -> str:
    """Strip the YAML front matter; the body is the human-facing instruction."""
    text = task_md.read_text(errors="replace")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.strip()


def solution(oracle: Path, budget: int) -> str:
    """Render the oracle directory as the answer a strong agent would write."""
    files = sorted(
        (p for p in oracle.rglob("*") if p.is_file() and p.suffix.lower() in CODE_SUFFIXES),
        # solve.sh is the entry point, so lead with it.
        key=lambda p: (p.name != "solve.sh", p.name),
    )
    if not files:
        return ""
    parts, used = [], 0
    for f in files:
        body = f.read_text(errors="replace").strip()
        if not body:
            continue
        # Oracle scripts carry scaffolding comments aimed at the harness
        # ("BenchFlow runs the oracle with...", issue references). They teach
        # the model about the grader rather than the task, so drop them.
        body = "\n".join(
            ln for ln in body.splitlines()
            if not re.match(r"^\s*#\s*(BenchFlow|#\d+|In oracle mode)", ln)
        )
        block = f"### `{f.relative_to(oracle)}`\n\n```{LANG.get(f.suffix.lower(), '')}\n{body}\n```"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    if not parts:
        return ""
    return ("Here is a complete solution.\n\n" + "\n\n".join(parts)
            + "\n\nRun `bash solve.sh` from `/app` to apply it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="task names scored by the eval -- these MUST be held out")
    ap.add_argument("--only", nargs="*", default=[],
                    help="build from ONLY these tasks. This deliberately inverts the "
                         "held-out rule and is for the leak control: train on the "
                         "evaluated tasks' own solutions so a positive delta is "
                         "guaranteed if the pipeline works at all. Never use for a "
                         "real submission -- it is contamination by construction.")
    ap.add_argument("--max-chars", type=int, default=14000,
                    help="cap per solution; very long oracles blow up sequence length")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="skip trivial solutions that teach nothing")
    args = ap.parse_args()

    excluded = set(args.exclude)
    only = set(args.only)
    if only and excluded:
        print("build: --only and --exclude are mutually exclusive", file=sys.stderr)
        return 2
    rows, skipped, held = [], [], 0
    for task in sorted(p for p in args.tasks_dir.iterdir() if p.is_dir()):
        if only:
            if task.name not in only:
                held += 1
                continue
        elif task.name in excluded:
            held += 1
            continue
        task_md, oracle = task / "task.md", task / "oracle"
        if not task_md.exists() or not oracle.is_dir():
            skipped.append((task.name, "no task.md/oracle"))
            continue
        prompt = instruction(task_md)
        answer = solution(oracle, args.max_chars)
        if not prompt or len(answer) < args.min_chars:
            skipped.append((task.name, "empty or trivial solution"))
            continue
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]})

    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    chars = [len(r["messages"][-1]["content"]) for r in rows]
    print(f"  wrote {len(rows)} rows -> {args.out}")
    print(f"  held out {held} eval tasks; skipped {len(skipped)}")
    for name, why in skipped[:5]:
        print(f"    skip {name}: {why}")
    if chars:
        chars.sort()
        print(f"  solution chars: median {chars[len(chars)//2]}, max {chars[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
