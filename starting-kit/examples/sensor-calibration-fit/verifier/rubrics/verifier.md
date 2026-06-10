# Sensor Calibration Fit Rubric

Single binary check: the verifier awards 1.0 when every pytest test in
`verifier/test_outputs.py` passes; otherwise 0.0.

A passing trial means `/root/results.json` exists, is valid JSON, and for
each sensor `S1`/`S2`/`S3` reports:

- `gain` and `offset` from an ordinary least squares fit of
  `reference ~ gain * raw + offset` on the calibration CSV **after**
  excluding every row where `|reference - raw| > 10.0` (only S2 has such
  rows — a 10-row stuck-output segment), within ±1e-4 and ±1e-3
  respectively of the values recomputed from the seed data;
- `rmse` of the fit residuals over the included rows (divide by n),
  within ±1e-3;
- `n_excluded` exactly equal to the recomputed exclusion count
  (S1: 0, S2: 10, S3: 0);
- `corrected` — one value per row of `field_readings.csv` for that
  sensor, in file order, each equal to `gain * raw + offset` within
  ±0.01.

The verifier recomputes all expected values from its own copy of the
seed CSVs in `verifier/data/`; a submission that fabricates numbers,
skips the outlier exclusion, inverts the regression, or tampers with
`/root/data` does not pass.
