---
document_version: '0.3'
verifier:
  name: sensor-calibration-fit-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      calibration_correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

A passing trial writes `/root/results.json` with one entry per sensor
(`S1`, `S2`, `S3`), each holding the OLS calibration `gain`/`offset`
fitted on the calibration CSV after excluding rows with
`|reference - raw| > 10.0` (S2 contains a 10-row stuck-sensor segment
that this rule removes), the fit `rmse` over the included rows, the
exact `n_excluded` count (0, 10, 0), and the corrected held-out field
readings (`gain * raw + offset`) in file order. The verifier recomputes
every expected value from its own pristine copy of the seed CSVs;
tolerances are ±1e-4 on gain, ±1e-3 on offset and rmse, ±0.01 on each
corrected value.
