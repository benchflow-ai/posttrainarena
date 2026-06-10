---
version: "1.0"
metadata:
  author_name: Xiangyi Li
  author_email: xiangyi@benchflow.ai
  category: media-content-production
  difficulty: medium
  tags: [subtitles, srt, quality-control, timing, captions, media]
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 300
environment:
  build_timeout_sec: 600
  cpus: 1
  memory_mb: 2048
  storage_mb: 4096
  allow_internet: false
---

## prompt

You are doing subtitle quality control on `/root/subtitles/input.srt`, a
SubRip (SRT) file with 120 cues. The file contains timing and numbering
defects. Produce two outputs: a machine-readable QC report listing every
defect in the original file, and a corrected SRT with the timing and
numbering repairs applied. Do not modify `/root/subtitles/input.srt`.

### Definitions

- A **cue** is one SRT block: an index line, a timing line
  `HH:MM:SS,mmm --> HH:MM:SS,mmm`, and one or more text lines.
- The **position** of a cue is its 1-based order of appearance in the
  file (first block = position 1). The **printed index** is the number
  on the cue's index line. Everywhere below, defects are reported by
  *position*, never by printed index.
- A cue's **duration** is `end - start` in milliseconds.
- A cue's **character count** is the total number of characters across
  its text lines, line breaks excluded (spaces and punctuation count).
- A cue's **CPS** (characters per second) is
  `character_count / (duration_ms / 1000.0)`.

### Defects to detect (in the original file, original order)

| `type` string            | Rule                                                                 |
|--------------------------|----------------------------------------------------------------------|
| `overlap`                | The cue's start is strictly earlier than the previous cue's end (compare against the cue immediately before it in file order; the first cue can never overlap). |
| `out_of_order_index`     | The cue's printed index is not equal to its position.                |
| `max_duration_exceeded`  | The cue's duration is strictly greater than 7000 ms.                 |
| `cps_exceeded`           | The cue's CPS is strictly greater than 17.0.                         |

A single cue can have more than one defect; report one entry per defect.

### Output 1 — QC report at `/root/outputs/qc_report.json`

A JSON object with exactly this shape:

```json
{
  "total_cues": 120,
  "counts": {
    "overlap": 0,
    "out_of_order_index": 0,
    "max_duration_exceeded": 0,
    "cps_exceeded": 0
  },
  "defects": [
    {"type": "overlap", "cue": 14}
  ]
}
```

- `total_cues`: number of cues in the input file.
- `counts`: all four keys, always present, each the number of defects of
  that type (the example shows zeros; report the real counts).
- `defects`: one object per defect with exactly the keys `type` (one of
  the four strings above) and `cue` (the 1-based position, an integer).
  Sort the array by `cue` ascending, then by `type` ascending
  (alphabetically) for entries on the same cue.

### Output 2 — corrected SRT at `/root/outputs/fixed.srt`

Apply exactly these repairs, in one pass over the cues in original file
order, keeping a running `prev_end` (the repaired end of the previous
cue):

1. **Renumber**: the printed index of every cue becomes its position
   (1, 2, ..., 120).
2. **De-overlap**: `new_start = max(original_start, prev_end)` (for the
   first cue, `new_start = original_start`).
3. **Clamp duration**: `new_end = original_end`; if
   `new_end - new_start > 7000` ms, set `new_end = new_start + 7000`.
4. Update `prev_end = new_end` and move to the next cue.

CPS violations are **report-only**: do not change any text and do not
change timing because of CPS. Text lines must be byte-for-byte
identical to the input. Do not reorder, merge, split, add, or delete
cues.

Format `fixed.srt` as standard SRT: for each cue, the index line, the
timing line `HH:MM:SS,mmm --> HH:MM:SS,mmm` (zero-padded, comma before
milliseconds), then the text lines; one blank line between cues; end
the file with a newline.

Write both files under `/root/outputs/` (create the directory). The
report must be valid JSON parseable with a standard JSON parser.
