#!/bin/bash
# Reference solution for sensor-calibration-fit. Runs inside the same
# container the agent uses. Pure stdlib: read the calibration CSVs, apply
# the stated |reference - raw| > 10.0 exclusion, fit OLS per sensor,
# correct the held-out field readings, write /root/results.json.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"
export BENCHFLOW_WORKSPACE="$WORKSPACE"

python3 - <<'PYTHON_SCRIPT'
import csv
import json
import math
import os
from pathlib import Path

workspace = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
data_dir = workspace / "data"
if not data_dir.is_dir():
    data_dir = Path("/root/data")

OUTLIER_THRESHOLD = 10.0
SENSORS = ["S1", "S2", "S3"]


def fit_sensor(sensor_id):
    with open(data_dir / f"calibration_{sensor_id}.csv", newline="") as f:
        rows = [(float(r["raw"]), float(r["reference"])) for r in csv.DictReader(f)]

    included = [(x, y) for x, y in rows if abs(y - x) <= OUTLIER_THRESHOLD]
    n_excluded = len(rows) - len(included)

    n = len(included)
    mean_x = sum(x for x, _ in included) / n
    mean_y = sum(y for _, y in included) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in included)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in included)
    gain = sxy / sxx
    offset = mean_y - gain * mean_x
    rmse = math.sqrt(
        sum((y - (gain * x + offset)) ** 2 for x, y in included) / n
    )
    return gain, offset, rmse, n_excluded


results = {"sensors": {}}
fits = {}
for sid in SENSORS:
    gain, offset, rmse, n_excluded = fit_sensor(sid)
    fits[sid] = (gain, offset)
    results["sensors"][sid] = {
        "gain": gain,
        "offset": offset,
        "rmse": rmse,
        "n_excluded": n_excluded,
        "corrected": [],
    }

with open(data_dir / "field_readings.csv", newline="") as f:
    for r in csv.DictReader(f):
        gain, offset = fits[r["sensor_id"]]
        results["sensors"][r["sensor_id"]]["corrected"].append(
            gain * float(r["raw"]) + offset
        )

out_path = workspace / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Oracle wrote {out_path}")
PYTHON_SCRIPT
