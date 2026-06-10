# Subtitle Overlap QC Rubric

Single binary check: the verifier awards 1.0 when every pytest test in
`verifier/test_outputs.py` passes, otherwise 0.0.

A passing trial:

- writes `/root/outputs/qc_report.json` — a JSON object with
  `total_cues: 120`, a `counts` object holding exactly the keys
  `overlap`, `out_of_order_index`, `max_duration_exceeded`,
  `cps_exceeded` with the true counts (9 / 7 / 7 / 11 for this seed),
  and a `defects` array with one `{"type": ..., "cue": ...}` entry per
  defect, `cue` being the 1-based position in the original file, sorted
  by cue ascending then type ascending;
- writes `/root/outputs/fixed.srt` — the input cues in original order,
  renumbered 1..120, with each cue's start clamped to
  `max(original_start, prev_repaired_end)` and each duration capped at
  7000 ms in a single forward pass; text lines byte-identical to the
  input; CPS violations not "fixed" (they are report-only).

The verifier recomputes the expected defect list and repaired timings
from a pristine seed copy inside the verifier package, so there is no
stored answer key to leak. Grading edge cases for a human reviewer:
extra or missing defect entries, mis-typed defect strings, cues
identified by printed index instead of position, unsorted defect
arrays, timing off by even 1 ms, altered text, or a `fixed.srt` that is
just the input copied back — all fail.
