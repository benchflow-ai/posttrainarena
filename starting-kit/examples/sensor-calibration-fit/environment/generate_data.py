#!/usr/bin/env python3
"""Deterministic seed-data generator for sensor-calibration-fit.

Run from this directory:

    python3 generate_data.py

Writes the calibration and field CSVs into ./data/ AND into
../verifier/data/ (the verifier recomputes all expected values from its
own pristine copy, so an agent that edits /root/data cannot move the
goalposts).

This script is provenance only — it is NOT copied into the task image
(the Dockerfile COPYs only data/), so the true gain/offset constants
below are never visible to the agent. Everything uses Python's stdlib
`random` with a fixed seed; reruns are byte-identical.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIRS = [HERE / "data", HERE.parent / "verifier" / "data"]

SEED = 20260610

# True linear drift per sensor: reference = gain * raw + offset + noise.
# n  = number of calibration rows
# sd = gaussian noise stdev on the reference reading
SENSORS = {
    "S1": {"gain": 1.0312, "offset": -1.85, "n": 40, "sd": 0.12},
    "S2": {"gain": 0.9685, "offset": 2.40, "n": 48, "sd": 0.15},
    "S3": {"gain": 1.0021, "offset": 0.75, "n": 36, "sd": 0.10},
}

# S2 has a stuck-sensor fault: rows 21..30 (1-based sample_id) report a
# constant raw value while the reference bath keeps sweeping. These rows
# violate |reference - raw| <= 10.0 by a wide margin (>= ~36 degC) and
# must be excluded by the stated rule before fitting.
FAULT_SENSOR = "S2"
FAULT_ROWS = range(20, 30)  # 0-based indices
FAULT_RAW = 85.0

RAW_LO, RAW_HI = 4.0, 48.0
FIELD_N = 12  # held-out raw readings per sensor


def gen_calibration(rng: random.Random) -> dict[str, list[tuple[int, float, float]]]:
    tables: dict[str, list[tuple[int, float, float]]] = {}
    for sid, cfg in SENSORS.items():
        rows = []
        n = cfg["n"]
        for i in range(n):
            sweep = RAW_LO + (RAW_HI - RAW_LO) * i / (n - 1)
            raw_true = round(sweep + rng.uniform(-0.4, 0.4), 3)
            reference = round(
                cfg["gain"] * raw_true + cfg["offset"] + rng.gauss(0.0, cfg["sd"]), 3
            )
            raw = raw_true
            if sid == FAULT_SENSOR and i in FAULT_ROWS:
                raw = FAULT_RAW
            rows.append((i + 1, raw, reference))
        tables[sid] = rows
    return tables


def gen_field(rng: random.Random) -> list[tuple[str, int, float]]:
    rows = []
    for sid in SENSORS:
        for j in range(FIELD_N):
            raw = round(rng.uniform(RAW_LO + 1.0, RAW_HI - 1.0), 3)
            rows.append((sid, j + 1, raw))
    return rows


def main() -> None:
    rng = random.Random(SEED)
    calibration = gen_calibration(rng)
    field = gen_field(rng)

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid, rows in calibration.items():
            with open(out_dir / f"calibration_{sid}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", "raw", "reference"])
                w.writerows(rows)
        with open(out_dir / "field_readings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sensor_id", "sample_id", "raw"])
            w.writerows(field)
        print(f"wrote {len(calibration)} calibration files + field_readings.csv -> {out_dir}")


if __name__ == "__main__":
    main()
