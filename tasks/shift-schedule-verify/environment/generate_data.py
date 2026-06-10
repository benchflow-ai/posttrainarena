#!/usr/bin/env python3
"""
Deterministic seed-data generator for the shift-schedule-verify task.

Run from this directory:  python3 generate_data.py

Writes:
  - instance.json          (the scheduling instance: workers, days, shifts,
                            availability matrix, rules)
  - broken_schedule.json   (a schedule with exactly one seeded violation of
                            each of the five violation types)
  - ../verifier/data/instance.json          (pristine copy for the verifier)
  - ../verifier/data/broken_schedule.json   (pristine copy for the verifier)

Everything is seeded (random.seed) so re-running reproduces identical bytes.
The generator constructs a known-valid base schedule FIRST, derives the
availability matrix as a superset of it (so a valid schedule is guaranteed
to exist), then perturbs a copy to create the broken schedule. The base
valid schedule is intentionally NOT written anywhere — the verifier checks
the agent's schedule by constraint-checking, not by comparison.
"""
import copy
import json
import random
from pathlib import Path

SEED = 20260610
random.seed(SEED)

WORKERS = ["amara", "boris", "chen", "divya", "elif",
           "farid", "gita", "hugo", "ines", "jonas"]
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
SHIFTS = ["opening", "midday", "closing"]
MIN_COVERAGE = {"opening": 2, "midday": 2, "closing": 2}
MAX_SHIFTS_PER_WEEK = 5
EXTRA_AVAILABILITY_P = 0.35


# ── canonical violation checker (mirrored in verifier/test_outputs.py) ──────
def compute_violations(instance, schedule):
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


# ── 1. base valid schedule (round-robin; provably no rule violations) ───────
base = {d: {s: [] for s in SHIFTS} for d in DAYS}
for di, d in enumerate(DAYS):
    for k in range(6):  # 2 workers per shift, 3 shifts
        w = WORKERS[(6 * di + k) % len(WORKERS)]
        base[d][SHIFTS[k // 2]].append(w)

# ── 2. availability = base schedule + seeded extra slots ────────────────────
availability = {w: {d: set() for d in DAYS} for w in WORKERS}
for d in DAYS:
    for s in SHIFTS:
        for w in base[d][s]:
            availability[w][d].add(s)
for w in WORKERS:
    for d in DAYS:
        for s in SHIFTS:
            if random.random() < EXTRA_AVAILABILITY_P:
                availability[w][d].add(s)

instance = {
    "workers": WORKERS,
    "days": DAYS,
    "shifts": SHIFTS,
    "availability": {
        w: {d: [s for s in SHIFTS if s in availability[w][d]] for d in DAYS}
        for w in WORKERS
    },
    "rules": {
        "max_shifts_per_day": 1,
        "max_shifts_per_week": MAX_SHIFTS_PER_WEEK,
        "min_coverage": MIN_COVERAGE,
        "no_closing_then_opening": True,
    },
}

assert not compute_violations(instance, base), "base schedule must be valid"


# ── 3. broken schedule: inject exactly one violation of each type ───────────
def loads(schedule):
    t = {}
    for d in DAYS:
        for s in SHIFTS:
            for w in set(schedule[d][s]):
                t[w] = t.get(w, 0) + 1
    return t


broken = copy.deepcopy(base)

# (a) under_coverage: drop the 2nd midday worker on tue (slot left with 1)
removed = broken["tue"]["midday"].pop(1)

# (b) unavailable: on thu closing, swap slot[0] for a worker who is NOT
#     available for thu/closing, works nowhere on thu, and stays under caps.
def pick_unavailable():
    t = loads(broken)
    on_thu = {w for s in SHIFTS for w in broken["thu"][s]}
    for w in WORKERS:
        if w in on_thu or t.get(w, 0) + 1 > MAX_SHIFTS_PER_WEEK:
            continue
        if "closing" in instance["availability"][w]["thu"]:
            continue
        if w in broken["fri"]["opening"]:  # avoid incidental close_to_open
            continue
        return w
    raise RuntimeError("no candidate for unavailable violation")

broken["thu"]["closing"][0] = pick_unavailable()

# (c) double_booked: add a worker already on mon to a 2nd mon shift where
#     they ARE available (no incidental unavailable violation).
def pick_double_book():
    t = loads(broken)
    for s_have in SHIFTS:
        for w in broken["mon"][s_have]:
            if t.get(w, 0) + 1 > MAX_SHIFTS_PER_WEEK:
                continue
            for s_add in SHIFTS:
                if s_add == s_have or w in broken["mon"][s_add]:
                    continue
                if s_add not in instance["availability"][w]["mon"]:
                    continue
                if s_add == "opening":  # mon is first day; safe anyway
                    pass
                if s_add == "closing" and w in broken["tue"]["opening"]:
                    continue
                return w, s_add
    raise RuntimeError("no candidate for double_booked violation")

w_db, s_db = pick_double_book()
broken["mon"][s_db].append(w_db)

# (d) over_max_shifts: push a worker to 6 assignments on a day off, where
#     available, without creating other violation types.
def pick_over_max():
    t = loads(broken)
    for w in WORKERS:
        if t.get(w, 0) != MAX_SHIFTS_PER_WEEK:
            continue
        for di, d in enumerate(DAYS):
            if any(w in broken[d][s] for s in SHIFTS):
                continue
            for s in SHIFTS:
                if s not in instance["availability"][w][d]:
                    continue
                if s == "opening" and di > 0 and w in broken[DAYS[di - 1]]["closing"]:
                    continue
                if s == "closing" and di < 6 and w in broken[DAYS[di + 1]]["opening"]:
                    continue
                return w, d, s
    raise RuntimeError("no candidate for over_max_shifts violation")

w_om, d_om, s_om = pick_over_max()
broken[d_om][s_om].append(w_om)

# (e) close_to_open: a worker who closes on some day also opens the next
#     day — where available, day off on the next day, within the weekly cap.
def pick_close_open():
    t = loads(broken)
    for i in range(len(DAYS) - 1, 0, -1):  # prefer late-week pairs
        d, nd = DAYS[i - 1], DAYS[i]
        for w in broken[d]["closing"]:
            if t.get(w, 0) + 1 > MAX_SHIFTS_PER_WEEK:
                continue
            if any(w in broken[nd][s] for s in SHIFTS):
                continue
            if "opening" not in instance["availability"][w][nd]:
                continue
            return w, nd
    raise RuntimeError("no candidate for close_to_open violation")

w_co, d_co = pick_close_open()
broken[d_co]["opening"].append(w_co)

# ── 4. sanity: the broken schedule has EXACTLY the five seeded violations ───
violations = compute_violations(instance, broken)
types = sorted(v[0] for v in violations)
assert types == sorted([
    "under_coverage", "unavailable", "double_booked",
    "over_max_shifts", "close_to_open",
]), f"unexpected violation set: {sorted(violations)}"

# ── 5. write artifacts ───────────────────────────────────────────────────────
here = Path(__file__).resolve().parent
verifier_data = here.parent / "verifier" / "data"
verifier_data.mkdir(parents=True, exist_ok=True)

instance_text = json.dumps(instance, indent=2) + "\n"
broken_text = json.dumps({"schedule": broken}, indent=2) + "\n"

(here / "instance.json").write_text(instance_text)
(here / "broken_schedule.json").write_text(broken_text)
(verifier_data / "instance.json").write_text(instance_text)
(verifier_data / "broken_schedule.json").write_text(broken_text)

print(f"removed for under_coverage: {removed}")
print("seeded violations:")
for v in sorted(violations):
    print(f"  {v}")
print("wrote instance.json + broken_schedule.json (env + verifier copies)")
