# Shift-schedule-verify Rubric

Single binary check: the verifier awards 1.0 when every pytest test in
`verifier/test_outputs.py` passes, otherwise 0.0.

A passing trial:

1. **Violations report** — `/root/violations.json` is a JSON object
   `{"violations": [...]}` whose entries, interpreted as canonical tuples,
   are exactly the set of violations the verifier recomputes from the
   pristine instance + broken schedule (kept in `verifier/data/`, not the
   workspace copies). Each entry has exactly the keys defined for its type
   in the prompt; values must be worker/day/shift ids from the instance; no
   duplicates; order does not matter.
2. **Repaired schedule** — `/root/schedule.json` is a JSON object
   `{"schedule": {...}}` with exactly the seven days, exactly the three
   shifts per day, distinct known worker ids per slot, and zero violations
   under the same constraint checker: availability, max one shift per
   worker per day, weekly cap, per-shift minimum coverage, and no
   closing-then-opening across consecutive days. Any fully valid schedule
   passes — correctness is decided by constraint-checking, not by
   comparison to a stored answer. Resubmitting the broken schedule
   verbatim fails an explicit test.

Edge cases for human grading: a report that is correct except for key
naming or extra keys fails mechanically by design (the prompt fixes the
exact schema); a schedule with surplus coverage above the minimum is
acceptable as long as every other rule holds.
