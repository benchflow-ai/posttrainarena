---
version: "1.0"
metadata:
  author_name: Xiangyi Li
  author_email: xiangyi@benchflow.ai
  category: industrial-physical-systems
  difficulty: medium
  task_type:
  - calculation
  modality:
  - csv
  - time-series
  interface:
  - python
  - terminal
  skill_type:
  - mathematical-method
  - data-cleaning-procedure
  tags:
  - sensors
  - calibration
  - least-squares
  - outlier-exclusion
  - csv
  - instrumentation
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  build_timeout_sec: 600
  cpus: 1
  memory_mb: 2048
  storage_mb: 10240
  allow_internet: false
---

## prompt

You are calibrating three industrial temperature sensors (`S1`, `S2`, `S3`)
against a reference instrument. For each sensor, fit a linear calibration
from bench data, apply it to held-out field readings, and write all results
to a single JSON file at `/root/results.json`.

### Input files

- `/root/data/calibration_S1.csv`, `/root/data/calibration_S2.csv`,
  `/root/data/calibration_S3.csv` — bench calibration runs, one per sensor.
  Columns: `sample_id` (int), `raw` (sensor reading, degC),
  `reference` (reference-instrument reading, degC).
- `/root/data/field_readings.csv` — held-out raw readings to correct.
  Columns: `sensor_id` (`S1`/`S2`/`S3`), `sample_id` (int), `raw` (degC).

### Procedure (follow exactly)

1. **Outlier exclusion.** For each sensor's calibration file, exclude every
   row where `|reference - raw| > 10.0`. (One sensor has a stuck-output
   fault segment; this rule removes it. Apply the same rule to all three
   sensors.) Use only the remaining rows for the fit. Record the number of
   excluded rows as `n_excluded`.
2. **Linear fit.** On the included rows, fit ordinary least squares
   coefficients `gain` and `offset` minimizing the sum of squared residuals
   of `reference - (gain * raw + offset)`. Equivalently:
   `gain = sum((raw - mean(raw)) * (ref - mean(ref))) / sum((raw - mean(raw))^2)`,
   `offset = mean(ref) - gain * mean(raw)`.
3. **Fit RMSE.** Over the included rows only, compute
   `rmse = sqrt(mean((reference - (gain * raw + offset))^2))`
   (divide by the number of included rows, not n-2).
4. **Apply the calibration.** For each row of `field_readings.csv`, compute
   `corrected = gain * raw + offset` using that row's sensor coefficients.

### Required output — `/root/results.json`

A JSON object with exactly this shape (report full float precision — at
least 6 significant digits; no NaN/Infinity):

```json
{
  "sensors": {
    "S1": {
      "gain": 1.0,
      "offset": 0.0,
      "rmse": 0.0,
      "n_excluded": 0,
      "corrected": [0.0]
    },
    "S2": { "...": "same keys as S1" },
    "S3": { "...": "same keys as S1" }
  }
}
```

- `corrected` lists the corrected values for that sensor's rows of
  `field_readings.csv`, in the same order those rows appear in the file
  (rows are grouped by sensor and sorted by `sample_id` within each sensor).
- `n_excluded` is an integer; the other numbers are JSON floats.

The verifier recomputes every expected value from the seed CSVs and checks:
`gain` within ±1e-4, `offset` and `rmse` within ±1e-3, each `corrected`
value within ±0.01, `n_excluded` exact, and the length of each `corrected`
array. Python 3.11 with `numpy` is available in the container; there is no
internet access.
