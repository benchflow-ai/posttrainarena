#!/bin/bash
# Reference solution. Runs inside the same container the agent uses.
# 1. Recomputes the typed violations report for the seeded broken schedule.
# 2. Builds a fully valid schedule by backtracking over (day, shift) slots.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"
export BENCHFLOW_WORKSPACE="$WORKSPACE"
PYTHON_BIN="${BENCHFLOW_PYTHON_BIN:-python3}"

"$PYTHON_BIN" - << 'PYTHON_SCRIPT'
import itertools
import json
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))

with open(WORKSPACE / "instance.json") as f:
    instance = json.load(f)
with open(WORKSPACE / "broken_schedule.json") as f:
    broken = json.load(f)["schedule"]

DAYS = instance["days"]
SHIFTS = instance["shifts"]
WORKERS = instance["workers"]
AVAIL = instance["availability"]
MAX_WEEK = instance["rules"]["max_shifts_per_week"]
MIN_COV = instance["rules"]["min_coverage"]


# ── Deliverable A: violations report (canonical checker) ────────────────────
def compute_violations(schedule):
    out = set()
    for d in DAYS:
        for s in SHIFTS:
            slot = schedule[d][s]
            for w in slot:
                if s not in AVAIL.get(w, {}).get(d, []):
                    out.add(("unavailable", w, d, s))
            if len(set(slot)) < MIN_COV[s]:
                out.add(("under_coverage", d, s))
    for d in DAYS:
        per_day = {}
        for s in SHIFTS:
            for w in set(schedule[d][s]):
                per_day[w] = per_day.get(w, 0) + 1
        for w, c in per_day.items():
            if c > 1:
                out.add(("double_booked", w, d))
    totals = {}
    for d in DAYS:
        for s in SHIFTS:
            for w in set(schedule[d][s]):
                totals[w] = totals.get(w, 0) + 1
    for w, c in totals.items():
        if c > MAX_WEEK:
            out.add(("over_max_shifts", w))
    for i in range(len(DAYS) - 1):
        d, nd = DAYS[i], DAYS[i + 1]
        for w in set(schedule[d]["closing"]):
            if w in schedule[nd]["opening"]:
                out.add(("close_to_open", w, nd))
    return out


KEYS = {
    "unavailable": ("worker", "day", "shift"),
    "double_booked": ("worker", "day"),
    "over_max_shifts": ("worker",),
    "close_to_open": ("worker", "day"),
    "under_coverage": ("day", "shift"),
}

violations = [
    {"type": t[0], **dict(zip(KEYS[t[0]], t[1:]))}
    for t in sorted(compute_violations(broken))
]
with open(WORKSPACE / "violations.json", "w") as f:
    json.dump({"violations": violations}, f, indent=2)
print(f"wrote violations.json ({len(violations)} violations)")


# ── Deliverable B: valid schedule via backtracking ───────────────────────────
SLOTS = [(d, s) for d in DAYS for s in SHIFTS]  # opening, midday, closing order


def solve():
    load = {w: 0 for w in WORKERS}
    schedule = {d: {s: [] for s in SHIFTS} for d in DAYS}

    def candidates(di, d, s, used_today):
        for w in WORKERS:
            if w in used_today:
                continue
            if s not in AVAIL[w][d]:
                continue
            if load[w] >= MAX_WEEK:
                continue
            if s == "opening" and di > 0 and w in schedule[DAYS[di - 1]]["closing"]:
                continue
            yield w

    def fill(idx):
        if idx == len(SLOTS):
            return True
        d, s = SLOTS[idx]
        di = DAYS.index(d)
        used_today = {w for sh in SHIFTS for w in schedule[d][sh]}
        cands = sorted(candidates(di, d, s, used_today), key=lambda w: (load[w], w))
        for combo in itertools.combinations(cands, MIN_COV[s]):
            schedule[d][s] = list(combo)
            for w in combo:
                load[w] += 1
            if fill(idx + 1):
                return True
            for w in combo:
                load[w] -= 1
            schedule[d][s] = []
        return False

    if not fill(0):
        raise SystemExit("no valid schedule found — instance should be feasible")
    return schedule


schedule = solve()
assert not compute_violations(schedule), "oracle schedule must be violation-free"
with open(WORKSPACE / "schedule.json", "w") as f:
    json.dump({"schedule": schedule}, f, indent=2)
print("wrote schedule.json (valid under all rules)")
PYTHON_SCRIPT

echo "Oracle complete."
