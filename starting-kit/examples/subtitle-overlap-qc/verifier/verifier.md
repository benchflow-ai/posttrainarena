---
document_version: '0.3'
verifier:
  name: subtitle-overlap-qc-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      qc_correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

A passing trial writes two files. `/root/outputs/qc_report.json` is a
JSON object whose `total_cues` is 120, whose `counts` object carries
exactly the four defect-type keys with the true per-type counts, and
whose `defects` array lists every defect in the seeded
`/root/subtitles/input.srt` — identified by 1-based file position and
one of the exact type strings `overlap`, `out_of_order_index`,
`max_duration_exceeded`, `cps_exceeded` — sorted by cue then type.
`/root/outputs/fixed.srt` is the input renumbered 1..120 with overlaps
clamped to the previous repaired end and durations capped at 7000 ms in
a single forward pass, text lines byte-identical to the input, and CPS
violations left untouched (report-only). The verifier recomputes both
expectations from a pristine copy of the seed shipped in
`verifier/data/input.srt`; submitting the input unchanged, omitting the
report, or any wrong/missing/spurious defect entry fails the trial.
