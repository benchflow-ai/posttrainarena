"""Verifier for subtitle-overlap-qc.

Recomputes the expected QC report and the expected repaired SRT from a
pristine copy of the seed file shipped inside the verifier package
(`verifier/data/input.srt`) — never from the (agent-writable) workspace
copy and never from a stored answer key. The agent's two outputs are
compared against those recomputed expectations.

Distinguishing power, by construction:
  - missing/empty output           -> TestOutputFilesExist fails
  - input copied back as fixed.srt -> timing/index tests fail (the seed
                                      contains real defects, so the
                                      expected repair differs from it)
  - fixed.srt only, no report      -> report tests fail
  - plausible-but-wrong report     -> set/order/count tests fail

Pytest passes -> reward 1.0; any failure -> 0.0 (see test.sh).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
FIXED_SRT = WORKSPACE / "outputs" / "fixed.srt"
REPORT_JSON = WORKSPACE / "outputs" / "qc_report.json"
SEED_SRT = Path(__file__).resolve().parent / "data" / "input.srt"

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


# --- reference implementation (mirrors the rules stated in task.md) --------

def parse_srt(text: str) -> list[dict]:
    """Parse SRT text into cues. Lenient about CRLF and trailing blanks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    cues = []
    for block in blocks:
        lines = [ln.rstrip() for ln in block.split("\n")]
        assert len(lines) >= 3, f"cue block too short: {lines!r}"
        index = int(lines[0].strip())
        m = TIMING_RE.match(lines[1])
        assert m, f"bad timing line: {lines[1]!r}"
        g = [int(x) for x in m.groups()]
        start = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        cues.append(
            {"index": index, "start": start, "end": end, "lines": lines[2:]}
        )
    return cues


def detect_defects(cues: list[dict]) -> list[dict]:
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


def repair(cues: list[dict]) -> list[dict]:
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


# --- expectations recomputed from the pristine seed ------------------------

def _expected():
    seed_cues = parse_srt(SEED_SRT.read_text(encoding="utf-8"))
    return seed_cues, detect_defects(seed_cues), repair(seed_cues)


SEED_CUES, EXPECTED_DEFECTS, EXPECTED_FIXED = _expected()


def load_report() -> dict:
    with open(REPORT_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_fixed_cues() -> list[dict]:
    return parse_srt(FIXED_SRT.read_text(encoding="utf-8"))


class TestOutputFilesExist:
    """Did the agent produce both deliverables at the stated paths?"""

    def test_fixed_srt_exists_nonempty(self):
        assert FIXED_SRT.exists(), f"missing {FIXED_SRT}"
        assert FIXED_SRT.stat().st_size > 0, f"{FIXED_SRT} is empty"

    def test_report_exists_valid_json_object(self):
        assert REPORT_JSON.exists(), f"missing {REPORT_JSON}"
        report = load_report()
        assert isinstance(report, dict), "qc_report.json must be a JSON object"


class TestQcReport:
    """Does the report match the defects recomputed from the seed file?"""

    def test_total_cues(self):
        assert load_report().get("total_cues") == len(SEED_CUES)

    def test_counts_object(self):
        counts = load_report().get("counts")
        assert isinstance(counts, dict), "'counts' must be an object"
        assert set(counts) == set(DEFECT_TYPES), (
            f"'counts' must have exactly the keys {sorted(DEFECT_TYPES)}, "
            f"got {sorted(counts)}"
        )
        expected = {
            t: sum(1 for d in EXPECTED_DEFECTS if d["type"] == t)
            for t in DEFECT_TYPES
        }
        assert counts == expected, f"expected counts {expected}, got {counts}"

    def test_defect_entries_shape(self):
        defects = load_report().get("defects")
        assert isinstance(defects, list), "'defects' must be an array"
        for d in defects:
            assert isinstance(d, dict) and set(d) == {"type", "cue"}, (
                f"each defect must have exactly keys type/cue, got {d!r}"
            )
            assert d["type"] in DEFECT_TYPES, f"unknown type {d['type']!r}"
            assert isinstance(d["cue"], int) and not isinstance(d["cue"], bool)
            assert 1 <= d["cue"] <= len(SEED_CUES)

    def test_defect_set_matches(self):
        got = sorted((d["cue"], d["type"]) for d in load_report()["defects"])
        want = sorted((d["cue"], d["type"]) for d in EXPECTED_DEFECTS)
        missing = [d for d in want if d not in got]
        extra = [d for d in got if d not in want]
        assert got == want, f"missing defects: {missing}; spurious: {extra}"

    def test_defects_sorted_as_specified(self):
        got = [(d["cue"], d["type"]) for d in load_report()["defects"]]
        want = [(d["cue"], d["type"]) for d in EXPECTED_DEFECTS]
        assert got == want, (
            "defects must be sorted by cue ascending, then type ascending"
        )


class TestFixedSrt:
    """Does fixed.srt match the repair recomputed from the seed file?"""

    def test_cue_count(self):
        assert len(load_fixed_cues()) == len(SEED_CUES)

    def test_indices_sequential(self):
        got = [c["index"] for c in load_fixed_cues()]
        assert got == list(range(1, len(SEED_CUES) + 1)), (
            "fixed.srt must be renumbered 1..N in file order"
        )

    def test_timings_repaired_exactly(self):
        got = load_fixed_cues()
        mismatches = [
            (pos, (g["start"], g["end"]), (e["start"], e["end"]))
            for pos, (g, e) in enumerate(zip(got, EXPECTED_FIXED), 1)
            if (g["start"], g["end"]) != (e["start"], e["end"])
        ]
        assert not mismatches, (
            "timing mismatches (pos, got(start,end), want(start,end)): "
            f"{mismatches[:10]}"
        )

    def test_text_lines_unchanged(self):
        got = load_fixed_cues()
        for pos, (g, e) in enumerate(zip(got, EXPECTED_FIXED), 1):
            assert g["lines"] == e["lines"], (
                f"cue {pos}: text lines must be byte-identical to the input"
            )

    def test_not_the_unmodified_input(self):
        """A copy of the defective input must not pass."""
        got = load_fixed_cues()
        seed_pairs = [(c["index"], c["start"], c["end"]) for c in SEED_CUES]
        got_pairs = [(c["index"], c["start"], c["end"]) for c in got]
        assert got_pairs != seed_pairs, (
            "fixed.srt is the unmodified input — no repairs were applied"
        )


class TestVerifierSelfConsistency:
    """Guards the verifier's own assumptions about the seed data."""

    def test_seed_has_defects_of_every_type(self):
        types = {d["type"] for d in EXPECTED_DEFECTS}
        assert types == set(DEFECT_TYPES)

    def test_expected_repair_differs_from_seed(self):
        seed_pairs = [(c["index"], c["start"], c["end"]) for c in SEED_CUES]
        fixed_pairs = [(c["index"], c["start"], c["end"]) for c in EXPECTED_FIXED]
        assert seed_pairs != fixed_pairs

    def test_expected_repair_is_clean(self):
        residual = [
            d for d in detect_defects(EXPECTED_FIXED)
            if d["type"] != "cps_exceeded"
        ]
        assert residual == [], (
            f"repair must clear every non-CPS defect, residual: {residual}"
        )
