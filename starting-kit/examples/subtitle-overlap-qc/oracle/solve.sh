#!/bin/bash
# Reference solution for subtitle-overlap-qc. Reads the seeded SRT,
# detects the four defect classes, writes the QC report, and writes the
# repaired SRT — exactly per the rules stated in task.md.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
export BENCHFLOW_WORKSPACE="$WORKSPACE"
mkdir -p "$WORKSPACE/outputs"

python3 - <<'PYTHON_SCRIPT'
import json
import os
import re
from pathlib import Path

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
INPUT_SRT = WORKSPACE / "subtitles" / "input.srt"
OUT_DIR = WORKSPACE / "outputs"

MAX_DURATION_MS = 7000
MAX_CPS = 17.0
DEFECT_TYPES = (
    "cps_exceeded",
    "max_duration_exceeded",
    "out_of_order_index",
    "overlap",
)

TIMING_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*$"
)


def parse_srt(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    cues = []
    for block in blocks:
        lines = block.split("\n")
        index = int(lines[0].strip())
        m = TIMING_RE.match(lines[1])
        g = [int(x) for x in m.groups()]
        start = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        cues.append({"index": index, "start": start, "end": end, "lines": lines[2:]})
    return cues


def detect_defects(cues):
    defects = []
    prev_end = None
    for pos, cue in enumerate(cues, 1):
        dur = cue["end"] - cue["start"]
        if cue["index"] != pos:
            defects.append({"type": "out_of_order_index", "cue": pos})
        if dur > MAX_DURATION_MS:
            defects.append({"type": "max_duration_exceeded", "cue": pos})
        chars = sum(len(ln) for ln in cue["lines"])
        if dur > 0 and chars * 1000.0 / dur > MAX_CPS:
            defects.append({"type": "cps_exceeded", "cue": pos})
        if prev_end is not None and cue["start"] < prev_end:
            defects.append({"type": "overlap", "cue": pos})
        prev_end = cue["end"]
    return sorted(defects, key=lambda d: (d["cue"], d["type"]))


def repair(cues):
    out = []
    prev_end = None
    for pos, cue in enumerate(cues, 1):
        start = cue["start"] if prev_end is None else max(cue["start"], prev_end)
        end = cue["end"]
        if end - start > MAX_DURATION_MS:
            end = start + MAX_DURATION_MS
        out.append({"index": pos, "start": start, "end": end, "lines": cue["lines"]})
        prev_end = end
    return out


def fmt_ts(ms):
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


cues = parse_srt(INPUT_SRT.read_text(encoding="utf-8"))
defects = detect_defects(cues)

report = {
    "total_cues": len(cues),
    "counts": {t: sum(1 for d in defects if d["type"] == t) for t in DEFECT_TYPES},
    "defects": defects,
}
(OUT_DIR / "qc_report.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)

blocks = []
for cue in repair(cues):
    blocks.append(
        f"{cue['index']}\n{fmt_ts(cue['start'])} --> {fmt_ts(cue['end'])}\n"
        + "\n".join(cue["lines"])
    )
(OUT_DIR / "fixed.srt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

print(f"Oracle wrote {OUT_DIR / 'qc_report.json'} and {OUT_DIR / 'fixed.srt'}")
print(f"{len(cues)} cues, {len(defects)} defects")
PYTHON_SCRIPT
