"""
Verifier tests for shift-schedule-verify.

Two deliverables are scored:
  A. /root/violations.json — a typed violations report for the seeded
     broken schedule. The expected set is RECOMPUTED here from pristine
     copies of the inputs (verifier/data/), never read from the image, so
     tampering with /root/instance.json or /root/broken_schedule.json in
     the workspace cannot change the expectations.
  B. /root/schedule.json — the agent's own schedule. It is verified by
     constraint-checking against the instance rules, not by comparing to a
     canonical answer: any fully valid schedule passes.

Pure standard library — no third-party imports beyond pytest.
"""

import json
import os
from pathlib import Path

import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
VIOLATIONS_FILE = WORKSPACE / "violations.json"
SCHEDULE_FILE = WORKSPACE / "schedule.json"

# Pristine inputs live next to this file; the agent never sees this copy.
DATA_DIR = Path(__file__).resolve().parent / "data"

VIOLATION_KEYS = {
    "unavailable": ("worker", "day", "shift"),
    "double_booked": ("worker", "day"),
    "over_max_shifts": ("worker",),
    "close_to_open": ("worker", "day"),
    "under_coverage": ("day", "shift"),
}


def load_instance():
    with open(DATA_DIR / "instance.json") as f:
        return json.load(f)


def load_broken_schedule():
    with open(DATA_DIR / "broken_schedule.json") as f:
        return json.load(f)["schedule"]


# ── canonical constraint checker (mirrors environment/generate_data.py) ─────
def compute_violations(instance, schedule):
    """Return the full violation set as canonical tuples."""
    days = instance["days"]
    shifts = instance["shifts"]
    avail = instance["availability"]
    max_week = instance["rules"]["max_shifts_per_week"]
    min_cov = instance["rules"]["min_coverage"]
    out = set()
    for d in days:
        for s in shifts:
            slot = schedule[d][s]
            for w in slot:
                if s not in avail.get(w, {}).get(d, []):
                    out.add(("unavailable", w, d, s))
            if len(set(slot)) < min_cov[s]:
                out.add(("under_coverage", d, s))
    for d in days:
        per_day = {}
        for s in shifts:
            for w in set(schedule[d][s]):
                per_day[w] = per_day.get(w, 0) + 1
        for w, c in per_day.items():
            if c > 1:
                out.add(("double_booked", w, d))
    totals = {}
    for d in days:
        for s in shifts:
            for w in set(schedule[d][s]):
                totals[w] = totals.get(w, 0) + 1
    for w, c in totals.items():
        if c > max_week:
            out.add(("over_max_shifts", w))
    for i in range(len(days) - 1):
        d, nd = days[i], days[i + 1]
        for w in set(schedule[d]["closing"]):
            if w in schedule[nd]["opening"]:
                out.add(("close_to_open", w, nd))
    return out


def canonical_tuple(entry):
    """Map a report entry (dict) to the canonical tuple for comparison."""
    vtype = entry["type"]
    return (vtype,) + tuple(entry[k] for k in VIOLATION_KEYS[vtype])


def format_tuples(tuples):
    return "\n".join(f"  {t}" for t in sorted(tuples)) or "  (none)"


# ── Deliverable A: violations report ─────────────────────────────────────────


class TestViolationsReport:
    def test_violations_file_exists(self):
        assert VIOLATIONS_FILE.exists(), (
            f"Violations report not found at {VIOLATIONS_FILE}"
        )

    def test_violations_file_is_valid_json_object(self):
        with open(VIOLATIONS_FILE) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"violations.json is not valid JSON: {e}")
        assert isinstance(data, dict), "violations.json must be a JSON object"
        assert "violations" in data, 'violations.json must have a "violations" key'
        assert isinstance(data["violations"], list), '"violations" must be a list'

    def test_violation_entries_well_formed(self):
        instance = load_instance()
        domains = {
            "worker": set(instance["workers"]),
            "day": set(instance["days"]),
            "shift": set(instance["shifts"]),
        }

        with open(VIOLATIONS_FILE) as f:
            entries = json.load(f)["violations"]
        assert entries, "violations list is empty — the broken schedule has violations"
        for i, entry in enumerate(entries):
            assert isinstance(entry, dict), f"violations[{i}] is not an object"
            vtype = entry.get("type")
            assert vtype in VIOLATION_KEYS, (
                f"violations[{i}] has unknown type {vtype!r}; "
                f"expected one of {sorted(VIOLATION_KEYS)}"
            )
            expected_keys = {"type", *VIOLATION_KEYS[vtype]}
            assert set(entry) == expected_keys, (
                f"violations[{i}] (type {vtype}) must have exactly the keys "
                f"{sorted(expected_keys)}, got {sorted(entry)}"
            )
            for key in VIOLATION_KEYS[vtype]:
                assert entry[key] in domains[key], (
                    f"violations[{i}].{key} = {entry[key]!r} is not a known "
                    f"{key} from instance.json"
                )

    def test_no_duplicate_violation_entries(self):
        with open(VIOLATIONS_FILE) as f:
            entries = json.load(f)["violations"]
        seen = [canonical_tuple(e) for e in entries]
        assert len(seen) == len(set(seen)), (
            "violations list contains duplicate entries"
        )

    def test_violations_match_recomputed_expected_set(self):
        instance = load_instance()
        broken = load_broken_schedule()
        expected = compute_violations(instance, broken)

        with open(VIOLATIONS_FILE) as f:
            entries = json.load(f)["violations"]
        reported = {canonical_tuple(e) for e in entries}

        missing = expected - reported
        extra = reported - expected
        assert reported == expected, (
            "Violations report does not match the recomputed ground truth.\n"
            f"Missing (real violations not reported):\n{format_tuples(missing)}\n"
            f"Extra (reported but not real):\n{format_tuples(extra)}"
        )


# ── Deliverable B: the agent's own schedule ──────────────────────────────────


class TestProposedSchedule:
    def test_schedule_file_exists(self):
        assert SCHEDULE_FILE.exists(), f"Schedule not found at {SCHEDULE_FILE}"

    def test_schedule_valid_json_shape(self):
        instance = load_instance()
        with open(SCHEDULE_FILE) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"schedule.json is not valid JSON: {e}")
        assert isinstance(data, dict) and "schedule" in data, (
            'schedule.json must be an object with a "schedule" key'
        )
        schedule = data["schedule"]
        assert isinstance(schedule, dict), '"schedule" must be an object'
        assert set(schedule) == set(instance["days"]), (
            f"schedule must have exactly the days {instance['days']}, "
            f"got {sorted(schedule)}"
        )
        workers = set(instance["workers"])
        for d in instance["days"]:
            day_obj = schedule[d]
            assert isinstance(day_obj, dict) and set(day_obj) == set(
                instance["shifts"]
            ), f"schedule[{d!r}] must have exactly the shifts {instance['shifts']}"
            for s in instance["shifts"]:
                slot = day_obj[s]
                assert isinstance(slot, list), f"schedule[{d!r}][{s!r}] must be a list"
                unknown = [w for w in slot if w not in workers]
                assert not unknown, (
                    f"schedule[{d!r}][{s!r}] contains unknown worker ids: {unknown}"
                )
                assert len(slot) == len(set(slot)), (
                    f"schedule[{d!r}][{s!r}] lists the same worker twice"
                )

    def test_schedule_is_not_the_broken_schedule(self):
        """A fixed-file resubmission of the seeded input must not pass."""
        broken = load_broken_schedule()
        with open(SCHEDULE_FILE) as f:
            schedule = json.load(f)["schedule"]

        def normalize(sch):
            return {
                d: {s: sorted(slot) for s, slot in day.items()}
                for d, day in sch.items()
            }

        assert normalize(schedule) != normalize(broken), (
            "schedule.json is just the broken schedule resubmitted; "
            "it must be a repaired, fully valid schedule"
        )

    def test_schedule_satisfies_all_constraints(self):
        instance = load_instance()
        with open(SCHEDULE_FILE) as f:
            schedule = json.load(f)["schedule"]
        violations = compute_violations(instance, schedule)
        assert not violations, (
            "Proposed schedule violates the instance rules:\n"
            f"{format_tuples(violations)}"
        )
