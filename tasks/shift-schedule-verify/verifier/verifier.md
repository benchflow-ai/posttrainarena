---
document_version: '0.3'
verifier:
  name: shift-schedule-verify-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      constraint_check_correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

A passing trial writes two files. `/root/violations.json` lists exactly the
violations present in the seeded broken schedule — each entry typed
(`unavailable`, `double_booked`, `over_max_shifts`, `close_to_open`,
`under_coverage`) and located with the keys the prompt specifies, no
duplicates, no extras, order irrelevant. `/root/schedule.json` is the
agent's own week schedule with all seven days and three shifts, only known
worker ids, and zero rule violations when checked against the instance:
availability respected, at most one shift per worker per day, weekly cap
respected, every slot at or above minimum coverage, and no closing shift
followed by an opening shift the next day. The schedule is judged purely by
constraint-checking against pristine copies of the inputs kept in
`verifier/data/` — there is no canonical answer, and resubmitting the
broken schedule unchanged fails.
