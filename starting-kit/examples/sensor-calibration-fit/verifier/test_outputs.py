"""Verifier for sensor-calibration-fit.

All expected values are RECOMPUTED here from the verifier's own pristine
copy of the seed CSVs (verifier/data/, shipped alongside this file).
Nothing is hardcoded, so:

- a missing/empty/garbage /root/results.json fails the existence tests;
- plausible-but-wrong numbers (e.g. fitting without the stated outlier
  exclusion, or regressing raw on reference) fail the numeric tests;
- a fixed file written without doing the work fails because the expected
  values derive from the data, and an agent editing /root/data cannot
  move the goalposts (this copy is authoritative).

Pure stdlib — no third-party imports beyond pytest.
"""

import csv
import json
import math
import os
from pathlib import Path

import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
RESULTS_FILE = WORKSPACE / "results.json"
DATA_DIR = Path(__file__).resolve().parent / "data"

SENSORS = ["S1", "S2", "S3"]
OUTLIER_THRESHOLD = 10.0

GAIN_TOL = 1e-4
OFFSET_TOL = 1e-3
RMSE_TOL = 1e-3
CORRECTED_TOL = 0.01

# Ground-truth structural fact about the seeded data: exactly one sensor
# (S2) has a 10-row fault segment. Asserted against the recomputation so a
# corrupted data copy fails loudly instead of validating garbage.
EXPECTED_FAULT_SENSOR = "S2"


def _fit_from_seed(sensor_id):
    """Recompute (gain, offset, rmse, n_excluded) from the pristine CSVs."""
    with open(DATA_DIR / f"calibration_{sensor_id}.csv", newline="") as f:
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
    rmse = math.sqrt(sum((y - (gain * x + offset)) ** 2 for x, y in included) / n)
    return gain, offset, rmse, n_excluded


def _expected():
    expected = {}
    for sid in SENSORS:
        gain, offset, rmse, n_excluded = _fit_from_seed(sid)
        expected[sid] = {
            "gain": gain,
            "offset": offset,
            "rmse": rmse,
            "n_excluded": n_excluded,
            "corrected": [],
        }
    with open(DATA_DIR / "field_readings.csv", newline="") as f:
        for r in csv.DictReader(f):
            sid = r["sensor_id"]
            expected[sid]["corrected"].append(
                expected[sid]["gain"] * float(r["raw"]) + expected[sid]["offset"]
            )
    return expected


EXPECTED = _expected()


def _load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


class TestSeedDataIntegrity:
    """Guard: the verifier's own data copy matches the task's design."""

    def test_seed_data_present(self):
        for sid in SENSORS:
            assert (DATA_DIR / f"calibration_{sid}.csv").exists()
        assert (DATA_DIR / "field_readings.csv").exists()

    def test_fault_segment_is_where_designed(self):
        for sid in SENSORS:
            n_excluded = EXPECTED[sid]["n_excluded"]
            if sid == EXPECTED_FAULT_SENSOR:
                assert n_excluded == 10, (
                    f"verifier seed copy corrupted: {sid} should have a "
                    f"10-row fault segment, found {n_excluded}"
                )
            else:
                assert n_excluded == 0, (
                    f"verifier seed copy corrupted: {sid} should have no "
                    f"excluded rows, found {n_excluded}"
                )


class TestResultsFile:
    """Did the agent write a valid output at the stated path?"""

    def test_file_exists(self):
        assert RESULTS_FILE.exists(), f"Results file not found at {RESULTS_FILE}"

    def test_file_not_empty(self):
        assert RESULTS_FILE.stat().st_size > 0, f"{RESULTS_FILE} is empty"

    def test_file_is_valid_json(self):
        try:
            _load_results()
        except json.JSONDecodeError as e:
            pytest.fail(f"{RESULTS_FILE} is not valid JSON: {e}")


class TestSchema:
    """Does the output match the schema the prompt specified?"""

    def test_has_sensors_object(self):
        data = _load_results()
        assert isinstance(data, dict), "results.json must be a JSON object"
        assert "sensors" in data, "missing required top-level key: 'sensors'"
        assert isinstance(data["sensors"], dict), "'sensors' must be an object"

    def test_all_three_sensors_present(self):
        sensors = _load_results()["sensors"]
        missing = [s for s in SENSORS if s not in sensors]
        assert not missing, f"missing sensor entries: {missing}"

    @pytest.mark.parametrize("sid", SENSORS)
    def test_sensor_entry_shape(self, sid):
        entry = _load_results()["sensors"][sid]
        for key in ("gain", "offset", "rmse"):
            assert key in entry, f"{sid}: missing key '{key}'"
            assert isinstance(entry[key], (int, float)) and not isinstance(
                entry[key], bool
            ), f"{sid}.{key} must be a number, got {entry[key]!r}"
            assert math.isfinite(entry[key]), f"{sid}.{key} must be finite"
        assert "n_excluded" in entry, f"{sid}: missing key 'n_excluded'"
        assert isinstance(entry["n_excluded"], int) and not isinstance(
            entry["n_excluded"], bool
        ), f"{sid}.n_excluded must be an integer, got {entry['n_excluded']!r}"
        assert "corrected" in entry, f"{sid}: missing key 'corrected'"
        assert isinstance(entry["corrected"], list), f"{sid}.corrected must be a list"

    @pytest.mark.parametrize("sid", SENSORS)
    def test_corrected_length(self, sid):
        entry = _load_results()["sensors"][sid]
        expected_n = len(EXPECTED[sid]["corrected"])
        assert len(entry["corrected"]) == expected_n, (
            f"{sid}.corrected has {len(entry['corrected'])} values, "
            f"expected {expected_n} (one per field_readings.csv row for {sid})"
        )


class TestCalibrationCoefficients:
    """OLS fit after the stated outlier exclusion, vs recomputed truth."""

    @pytest.mark.parametrize("sid", SENSORS)
    def test_n_excluded(self, sid):
        entry = _load_results()["sensors"][sid]
        exp = EXPECTED[sid]["n_excluded"]
        assert entry["n_excluded"] == exp, (
            f"{sid}.n_excluded: expected {exp}, got {entry['n_excluded']} — "
            f"apply the |reference - raw| > {OUTLIER_THRESHOLD} row exclusion"
        )

    @pytest.mark.parametrize("sid", SENSORS)
    def test_gain(self, sid):
        entry = _load_results()["sensors"][sid]
        exp = EXPECTED[sid]["gain"]
        assert abs(entry["gain"] - exp) <= GAIN_TOL, (
            f"{sid}.gain: expected {exp:.6f} (±{GAIN_TOL}), got {entry['gain']}"
        )

    @pytest.mark.parametrize("sid", SENSORS)
    def test_offset(self, sid):
        entry = _load_results()["sensors"][sid]
        exp = EXPECTED[sid]["offset"]
        assert abs(entry["offset"] - exp) <= OFFSET_TOL, (
            f"{sid}.offset: expected {exp:.6f} (±{OFFSET_TOL}), got {entry['offset']}"
        )

    @pytest.mark.parametrize("sid", SENSORS)
    def test_rmse(self, sid):
        entry = _load_results()["sensors"][sid]
        exp = EXPECTED[sid]["rmse"]
        assert abs(entry["rmse"] - exp) <= RMSE_TOL, (
            f"{sid}.rmse: expected {exp:.6f} (±{RMSE_TOL}), got {entry['rmse']} — "
            f"RMSE is over included rows only, divided by n"
        )


class TestCorrectedFieldReadings:
    """Held-out series corrected with the fitted coefficients."""

    @pytest.mark.parametrize("sid", SENSORS)
    def test_corrected_values(self, sid):
        entry = _load_results()["sensors"][sid]
        exp_list = EXPECTED[sid]["corrected"]
        errors = []
        for i, (got, exp) in enumerate(zip(entry["corrected"], exp_list)):
            if (
                not isinstance(got, (int, float))
                or isinstance(got, bool)
                or not math.isfinite(got)
                or abs(got - exp) > CORRECTED_TOL
            ):
                errors.append(
                    f"  corrected[{i}]: expected {exp:.4f} (±{CORRECTED_TOL}), got {got!r}"
                )
        assert not errors, f"{sid} corrected-value mismatches:\n" + "\n".join(errors)
