#!/usr/bin/env python3
"""Regenerate assistant targets in a chat JSONL with a teacher model.

Keeps every row's metadata and user prompt; replaces the assistant turn with
the teacher's completion via any OpenAI-compatible chat endpoint. Used to
build teacher-distilled SFT corpora comparable row-for-row with the oracle
corpus they came from.

    TEACHER_API_KEY=sk-... python3 scripts/teacher_generate.py \
        --in sb-oracle-train-76.jsonl --out sb-ds4pro-train-76.jsonl \
        --base-url https://api.deepseek.com/v1 --model deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "You are an expert engineer. Produce the complete, self-contained solution "
    "to the task. Present every solution file as a fenced code block preceded "
    "by its backticked filename, with `solve.sh` as the entrypoint first."
)


def complete(base_url: str, model: str, key: str, messages: list[dict],
             max_tokens: int, attempts: int = 3) -> str:
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "posttrain-arena-teacher/1.0"})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                choice = json.load(r)["choices"][0]
            text = choice["message"]["content"]
            if not text:
                # Reasoning models return an empty `content` when max_tokens is
                # spent inside the hidden reasoning trace.
                raise RuntimeError(
                    f"empty content (finish_reason={choice.get('finish_reason')}); "
                    "raise --max-tokens")
            return text
        except Exception as exc:
            if i == attempts - 1:
                raise RuntimeError(f"teacher call failed after {attempts} tries: {exc}")
            time.sleep(5 * (i + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = os.environ.get("TEACHER_API_KEY")
    if not key:
        print("set TEACHER_API_KEY", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.src.read_text().splitlines() if l.strip()]

    # Resume: rows already generated (non-empty assistant turn) are kept as-is,
    # and each new success is appended immediately so a crash never loses work.
    done: dict[str, dict] = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["messages"][-1]["role"] == "assistant" and r["messages"][-1]["content"]:
                    done[r["task_id"]] = r
    todo = [r for r in rows if r["task_id"] not in done]
    print(f"{len(done)} rows already done, generating {len(todo)}")

    lock = threading.Lock()
    failures: list[str] = []

    def one(row: dict) -> None:
        prompt = [m for m in row["messages"] if m["role"] != "assistant"]
        try:
            text = complete(args.base_url, args.model, key,
                            [{"role": "system", "content": SYSTEM}, *prompt],
                            args.max_tokens)
        except Exception as exc:
            print(f"  {row['task_id']}: FAILED ({exc})", flush=True)
            with lock:
                failures.append(row["task_id"])
            return
        out = dict(row)
        out["messages"] = [*prompt, {"role": "assistant", "content": text}]
        out["teacher"] = args.model
        with lock:
            done[row["task_id"]] = out
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  {row['task_id']}: {len(text)} chars", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))

    # Final rewrite: source order, deduplicated.
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            if r["task_id"] in done:
                f.write(json.dumps(done[r["task_id"]], ensure_ascii=False) + "\n")
    if failures:
        print(f"FAILED {len(failures)} tasks: {' '.join(sorted(failures))}")
        return 1
    print(f"wrote {len(done)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
