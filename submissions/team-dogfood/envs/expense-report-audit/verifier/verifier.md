---
document_version: '0.3'
verifier:
  name: expense-report-audit-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      audit_correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

A passing trial writes `/root/audit.json` whose every value is the correct
mechanical application of the reimbursement policy to `/root/expenses.csv`:
`total_submitted` (sum of all line amounts) and `total_reimbursable` (sum of
the lines that survive the policy), one `line_items` entry per CSV row in
`row_id` order carrying the row's `reimbursable` verdict and its `flags`
(`unknown_category`, `missing_receipt`, or `over_cap`), the `flag_counts`
totals, and `policy_compliant` (true only when no line is flagged). The
verifier recomputes every expected value from its own pristine copy of the
seed CSV (`verifier/data/expenses.csv`); money is compared to within half a
cent, and the per-day caps are applied greedily in `row_id` order exactly as
stated in `task.md`. A submission that fabricates totals, skips the cap or
receipt rules, or reorders the rows does not pass.
