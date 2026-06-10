#!/usr/bin/env python3
"""Authoring-time generator for the subtitle-overlap-qc seed data.

Run from the task directory:

    python3 environment/generate_seed.py

Writes:
    environment/media/input.srt   (copied into the image by the Dockerfile)
    verifier/data/input.srt       (pristine copy the verifier recomputes from)

Deterministic: a fixed RNG seed, no wall-clock input, no environment input.
This script is intentionally NOT copied into the task image — the agent only
ever sees the generated .srt. Re-running this script must be a no-op diff.

The defect rules here mirror, exactly, the rules stated in task.md and
re-implemented in verifier/test_outputs.py and oracle/solve.sh:

  - overlap:                cue start < previous cue end (file order)
  - out_of_order_index:     printed index != 1-based file position
  - max_duration_exceeded:  (end - start) > 7000 ms
  - cps_exceeded:           chars / duration_seconds > 17.0, where chars is
                            the total character count of the text lines
                            (line breaks excluded)
"""
from __future__ import annotations

import random
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent

N_CUES = 120
SEED = 20260610

MAX_DURATION_MS = 7000
MAX_CPS = 17.0

N_OVERLAP = 9
N_TOO_LONG = 7
N_CPS = 11
N_INDEX_SWAPS = 3  # each swap makes 2 defective cues; +1 stale index = 7 total

WORDS = (
    "the quick seaside light drifts over salt water while gulls trace slow "
    "circles above rust colored hulls and the evening ferry hums against "
    "its moorings as lanterns flicker along the pier old nets dry beside "
    "crates of glass floats and somewhere a radio plays a tune nobody "
    "remembers learning"
).split()


def fmt_ts(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


def make_lines(rng: random.Random, target_chars: int, at_least: bool) -> list[str]:
    """Assemble words into 1-2 text lines.

    at_least=False: stop before exceeding target_chars (normal / low CPS).
    at_least=True:  keep adding until reaching target_chars (high CPS).
    """
    words: list[str] = []
    count = 0
    while True:
        w = rng.choice(WORDS)
        add = len(w) + (1 if words else 0)
        if at_least:
            words.append(w)
            count += add
            if count >= target_chars:
                break
        else:
            if words and count + add > target_chars - 1:  # reserve 1 for "."
                break
            words.append(w)
            count += add
    text = " ".join(words)
    text = text[0].upper() + text[1:] + "."
    if len(text) <= 42:
        return [text]
    # split at the word boundary closest to the midpoint
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    split_at = min(spaces, key=lambda i: abs(i - mid))
    return [text[:split_at], text[split_at + 1:]]


def char_count(lines: list[str]) -> int:
    return sum(len(line) for line in lines)


def main() -> None:
    rng = random.Random(SEED)

    # ---- defect plan ------------------------------------------------------
    timing_pool = list(range(2, N_CUES + 1))  # overlap needs a predecessor
    rng.shuffle(timing_pool)
    overlap_pos = set(timing_pool[:N_OVERLAP])
    too_long_pos = set(timing_pool[N_OVERLAP:N_OVERLAP + N_TOO_LONG])
    cps_pos = set(timing_pool[N_OVERLAP + N_TOO_LONG:N_OVERLAP + N_TOO_LONG + N_CPS])

    defect_positions = overlap_pos | too_long_pos | cps_pos
    swap_candidates = [
        p for p in range(2, N_CUES)
        if p not in defect_positions and (p + 1) not in defect_positions
    ]
    rng.shuffle(swap_candidates)
    swaps: list[int] = []
    used: set[int] = set()
    for p in swap_candidates:
        if p in used or (p + 1) in used:
            continue
        swaps.append(p)
        used.update({p - 1, p, p + 1, p + 2})
        if len(swaps) == N_INDEX_SWAPS:
            break
    assert len(swaps) == N_INDEX_SWAPS

    # one cue carries two defects on purpose: cps_exceeded + out_of_order_index
    stale_q = sorted(cps_pos)[0]

    printed = {i: i for i in range(1, N_CUES + 1)}
    for p in swaps:
        printed[p], printed[p + 1] = p + 1, p
    printed[stale_q] = stale_q + 57

    # ---- build cues -------------------------------------------------------
    cues = []  # (printed_index, start_ms, end_ms, lines)
    prev_end = 0
    t = 1000
    for pos in range(1, N_CUES + 1):
        gap = rng.randint(300, 1500)
        if pos in overlap_pos:
            delta = rng.randint(300, 900)
            start = prev_end - delta
            dur = rng.randint(2500, 6000)
            cps_target = rng.uniform(9.0, 14.0)
            lines = make_lines(rng, int(cps_target * dur / 1000), at_least=False)
        elif pos in too_long_pos:
            start = prev_end + gap if pos > 1 else t
            dur = rng.randint(7600, 9600)
            cps_target = rng.uniform(6.0, 10.0)
            lines = make_lines(rng, min(84, int(cps_target * dur / 1000)), at_least=False)
        elif pos in cps_pos:
            start = prev_end + gap if pos > 1 else t
            dur = rng.randint(1200, 2200)
            lines = make_lines(rng, int(19.5 * dur / 1000) + 1, at_least=True)
        else:
            start = prev_end + gap if pos > 1 else t
            dur = rng.randint(1500, 6000)
            cps_target = rng.uniform(9.0, 14.0)
            lines = make_lines(rng, max(14, int(cps_target * dur / 1000)), at_least=False)
        end = start + dur
        cues.append((printed[pos], start, end, lines))
        prev_end = end

    # ---- self-check against the stated rules ------------------------------
    found = {"overlap": [], "out_of_order_index": [],
             "max_duration_exceeded": [], "cps_exceeded": []}
    prev_end = None
    for pos, (idx, start, end, lines) in enumerate(cues, 1):
        dur = end - start
        assert dur > 0
        if idx != pos:
            found["out_of_order_index"].append(pos)
        if dur > MAX_DURATION_MS:
            found["max_duration_exceeded"].append(pos)
        cps = char_count(lines) * 1000.0 / dur
        if cps > MAX_CPS:
            found["cps_exceeded"].append(pos)
        if prev_end is not None and start < prev_end:
            found["overlap"].append(pos)
            assert end >= prev_end + 1500, f"degenerate overlap at {pos}"
        # margins: nothing may sit near the CPS boundary
        assert cps <= 15.5 or cps >= 18.0, f"cue {pos} too close to CPS limit: {cps:.2f}"
        # nothing may sit near the duration boundary
        assert dur <= MAX_DURATION_MS or dur >= 7600, f"cue {pos} near duration limit"
        prev_end = end

    assert sorted(found["overlap"]) == sorted(overlap_pos)
    assert sorted(found["max_duration_exceeded"]) == sorted(too_long_pos)
    assert sorted(found["cps_exceeded"]) == sorted(cps_pos)
    assert len(found["out_of_order_index"]) == 2 * N_INDEX_SWAPS + 1
    assert stale_q in found["cps_exceeded"] and stale_q in found["out_of_order_index"]

    # ---- write ------------------------------------------------------------
    blocks = []
    for idx, start, end, lines in cues:
        blocks.append(f"{idx}\n{fmt_ts(start)} --> {fmt_ts(end)}\n" + "\n".join(lines))
    text = "\n\n".join(blocks) + "\n"

    for out in (TASK_DIR / "environment" / "media" / "input.srt",
                TASK_DIR / "verifier" / "data" / "input.srt"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")

    total = sum(len(v) for v in found.values())
    print(f"{N_CUES} cues, {total} defects: " +
          ", ".join(f"{k}={len(v)}" for k, v in sorted(found.items())))


if __name__ == "__main__":
    main()
