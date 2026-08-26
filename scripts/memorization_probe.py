#!/usr/bin/env python3
"""Did the fine-tune actually learn? Measure memorization, not eval score.

An overfit proof-of-concept asks a narrower question than a benchmark does:
given a tiny corpus and many epochs, *did the weights move at all*. Job state
COMPLETED does not answer that -- it only says the job ran. A benchmark score
does not answer it either, because overfitting is deliberately the opposite of
generalization.

The answer is memorization. Replay each training prompt against the tuned model
and against its own base, and score the response against the training target.
A tuned model that has learned reproduces its targets far better than base.

The held-out control is what makes this evidence. Prompts the model never saw
should show little or no lift. If held-out lift matches training lift, the
"memorization" is really style transfer or a scoring artifact, and the probe
has proved nothing.

    FIREWORKS_API_KEY=fw_... python3 scripts/memorization_probe.py \\
        --train sft_lhtb.jsonl --holdout sft_lhtb_unfiltered.jsonl \\
        --base accounts/fireworks/models/qwen3p6-27b \\
        --tuned accounts/<your-account>/models/<your-tuned-model>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
_WORD = re.compile(r"[A-Za-z0-9_./-]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    counts: dict[str, int] = {}
    for t in g:
        counts[t] = counts.get(t, 0) + 1
    overlap = 0
    for t in p:
        if counts.get(t, 0) > 0:
            counts[t] -= 1
            overlap += 1
    if not overlap:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def complete(model: str, messages: list[dict], key: str, max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        # Greedy. Sampling noise would sit on top of the very effect being measured.
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Cloudflare fronts the inference API and 403s ("error code: 1010")
            # on urllib's default User-Agent, so send an explicit one.
            "User-Agent": "posttrain-arena-probe/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        return d["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as exc:
        return f"__ERROR__ HTTP {exc.code} {exc.read()[:180]!r}"
    except Exception as exc:
        return f"__ERROR__ {exc}"


def load(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line).get("messages") or []
        if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            continue
        rows.append({"prompt": msgs[:-1], "target": msgs[-1].get("content") or ""})
        if limit and len(rows) >= limit:
            break
    return rows


def score_set(name, rows, base, tuned, key, max_tokens, workers):
    def one(i_row):
        i, row = i_row
        b = complete(base, row["prompt"], key, max_tokens)
        t = complete(tuned, row["prompt"], key, max_tokens)
        return {
            "i": i,
            "base_f1": token_f1(b, row["target"]),
            "tuned_f1": token_f1(t, row["target"]),
            "base_err": b.startswith("__ERROR__"),
            "tuned_err": t.startswith("__ERROR__"),
            "base_msg": b[:150] if b.startswith("__ERROR__") else "",
            "tuned_msg": t[:150] if t.startswith("__ERROR__") else "",
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, enumerate(rows)))
    ok = [r for r in out if not r["base_err"] and not r["tuned_err"]]
    errs = [r for r in out if r["base_err"] or r["tuned_err"]]
    print(f"\n  {name}: {len(ok)}/{len(out)} scored"
          + (f"  ({len(errs)} errored)" if errs else ""))
    for r in errs[:3]:
        print(f"      error: {r['base_msg'] or r['tuned_msg']}")
    if not ok:
        return None
    mb = sum(r["base_f1"] for r in ok) / len(ok)
    mt = sum(r["tuned_f1"] for r in ok) / len(ok)
    wins = sum(1 for r in ok if r["tuned_f1"] > r["base_f1"])
    print(f"      base  mean token-F1 {mb:.4f}")
    print(f"      tuned mean token-F1 {mt:.4f}")
    print(f"      lift  {mt - mb:+.4f}   tuned wins {wins}/{len(ok)} rows")
    return {"n": len(ok), "base": mb, "tuned": mt, "lift": mt - mb, "wins": wins}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path, help="corpus the tuned model saw")
    ap.add_argument("--holdout", type=Path, help="corpus it did NOT see (the control)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("probe: set FIREWORKS_API_KEY", file=sys.stderr)
        return 2

    print(f"  base  {args.base}")
    print(f"  tuned {args.tuned}")
    train = load(args.train, args.limit)

    # Recall -- and therefore F1 -- is hard-capped at max_tokens/len(target). A
    # generation budget well below the target length makes a perfect memorizer
    # look like a failed run, so refuse to report a verdict off such a run.
    if train:
        lens = sorted(len(tokens(r["target"])) for r in train)
        median = lens[len(lens) // 2]
        ceiling = min(1.0, args.max_tokens / median)
        print(f"  target tokens: median {median}, max {lens[-1]}; "
              f"--max-tokens {args.max_tokens} caps recall at ~{ceiling:.2f}")
        if ceiling < 0.5:
            print(f"\n  ABORT: with only {args.max_tokens} generated tokens against a "
                  f"median target of {median}, even perfect memorization would score "
                  f"F1 <= ~{2 * ceiling / (1 + ceiling):.3f}. Raise --max-tokens to at "
                  f"least {median} or point --train at a length-capped corpus.")
            return 3
    res = {"train": score_set("TRAIN (seen)", train, args.base, args.tuned, key,
                              args.max_tokens, args.workers)}

    if args.holdout:
        seen = {json.dumps(r["prompt"], sort_keys=True) for r in train}
        held = [r for r in load(args.holdout, None)
                if json.dumps(r["prompt"], sort_keys=True) not in seen][: args.limit]
        if held:
            res["holdout"] = score_set("HELD-OUT (control)", held, args.base, args.tuned,
                                       key, args.max_tokens, args.workers)
        else:
            print("\n  HELD-OUT: no rows left after removing seen prompts")

    t, h = res.get("train"), res.get("holdout")
    print("\n  VERDICT")
    if not t:
        print("      inconclusive -- nothing scored")
    elif t["lift"] <= 0.02:
        print(f"      NO memorization (train lift {t['lift']:+.4f}). The tuned weights "
              "are not reproducing their own training targets; treat the run as not learned.")
    elif h and h["lift"] >= t["lift"] * 0.7:
        print(f"      AMBIGUOUS -- train lift {t['lift']:+.4f} but held-out lift "
              f"{h['lift']:+.4f} is nearly as large, so this is not specific to the "
              "training rows. Suspect style transfer or a scoring artifact.")
    else:
        ctrl = f"vs held-out {h['lift']:+.4f}" if h else "(no control run)"
        print(f"      MEMORIZATION CONFIRMED -- train lift {t['lift']:+.4f} {ctrl}. "
              "Gradients reached the weights and the adapter is live at inference.")

    if args.json_out:
        args.json_out.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
