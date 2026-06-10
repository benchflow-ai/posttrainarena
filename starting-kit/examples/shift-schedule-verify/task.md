---
version: "1.0"
metadata:
  author_name: Xiangyi Li
  author_email: xiangyi@benchflow.ai
  category: mathematics-or-formal-reasoning
  difficulty: medium
  task_type:
  - verification
  - planning
  modality:
  - json
  interface:
  - python
  - terminal
  skill_type:
  - mathematical-method
  tags:
  - constraint-satisfaction
  - scheduling
  - formal-reasoning
  - json
  - verification
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  build_timeout_sec: 600
  cpus: 1
  memory_mb: 1024
  storage_mb: 4096
  allow_internet: false
---

## prompt

You are auditing and repairing a one-week shift schedule for a small shop.
Two input files are in `/root`:

- `/root/instance.json` — the scheduling instance:
  - `workers`: list of worker ids.
  - `days`: the seven days in order (`mon` … `sun`). Days are consecutive;
    the week does **not** wrap around (`sun` closing followed by `mon`
    opening is fine).
  - `shifts`: the three shifts in order (`opening`, `midday`, `closing`).
  - `availability`: `availability[worker][day]` is the list of shifts that
    worker is able to work on that day.
  - `rules`:
    - `max_shifts_per_day: 1` — a worker may work at most one shift per day.
    - `max_shifts_per_week` — maximum total assignments per worker across
      the week.
    - `min_coverage` — for each shift, the minimum number of distinct
      workers that must be assigned to it on every day.
    - `no_closing_then_opening: true` — a worker who works `closing` on one
      day must not work `opening` on the immediately following day.
- `/root/broken_schedule.json` — a proposed schedule that violates some
  rules. Shape: `{"schedule": {<day>: {<shift>: [<worker>, ...]}}}` with
  every day and every shift present. Workers within a slot are distinct.

Produce two output files.

### Deliverable A — `/root/violations.json`

A complete violations report for the broken schedule, as a JSON object
`{"violations": [...]}`. Each element is an object of one of these five
types, with **exactly** the keys shown (all values are strings taken from
`instance.json`):

1. `{"type": "unavailable", "worker": W, "day": D, "shift": S}` — one entry
   for each assignment where worker `W` is listed in slot (`D`, `S`) but `S`
   is not in `availability[W][D]`.
2. `{"type": "double_booked", "worker": W, "day": D}` — one entry per
   (worker, day) where `W` is assigned to more than one shift on day `D`
   (a single entry regardless of how many shifts).
3. `{"type": "over_max_shifts", "worker": W}` — one entry per worker whose
   total number of assignments across the week exceeds
   `max_shifts_per_week`. Each slot a worker appears in counts as one
   assignment.
4. `{"type": "close_to_open", "worker": W, "day": D}` — one entry for each
   case where `W` works `closing` on some day and `opening` on the next
   day; `D` is the day of the **opening** shift.
5. `{"type": "under_coverage", "day": D, "shift": S}` — one entry per slot
   (`D`, `S`) whose number of assigned workers is below
   `min_coverage[S]`.

Report rules: include every violation that the broken schedule actually
contains and nothing else; no duplicate entries; list order does not
matter. Rules are checked independently — one violation does not suppress
another (for example, a double-booked assignment still counts toward the
worker's weekly total).

### Deliverable B — `/root/schedule.json`

Your own replacement schedule, in the same shape as
`broken_schedule.json`: `{"schedule": {<day>: {<shift>: [<worker>, ...]}}}`
with exactly the seven days and exactly the three shifts per day. Every
slot must list distinct worker ids from `instance.json`. The schedule must
satisfy **all** the rules: every assigned worker is available for that
slot, no worker works more than one shift per day, no worker exceeds
`max_shifts_per_week`, every slot has at least `min_coverage[shift]`
workers, and no worker works `closing` followed by `opening` the next day.

A valid schedule is guaranteed to exist. The verifier checks your schedule
mechanically against the constraints — any fully valid schedule passes;
there is no single expected answer. Do not modify the input files: the
verifier recomputes everything from pristine copies, so editing
`/root/instance.json` or `/root/broken_schedule.json` cannot change the
expected results.
