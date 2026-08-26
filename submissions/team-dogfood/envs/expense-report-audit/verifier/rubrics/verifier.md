# Expense Report Policy Audit Rubric

Single binary check: the verifier awards 1.0 when every pytest test in
`verifier/test_outputs.py` passes, and 0.0 otherwise.

A passing trial means `/root/audit.json` exists, is valid JSON, and matches
the ground truth recomputed from the verifier's pristine copy of the seed
CSV (`verifier/data/expenses.csv`). Concretely:

- the top-level object has exactly the keys `total_submitted`,
  `total_reimbursable`, `line_items`, `flag_counts`, `policy_compliant`,
  with the documented JSON types (numbers for money, integers for counts, a
  real boolean for `policy_compliant`);
- `total_submitted` equals the sum of every row's amount and
  `total_reimbursable` equals the sum of the amounts of the lines that are
  reimbursable, each within half a cent;
- `line_items` has one entry per CSV row, in `row_id` order, and each entry's
  `category`, `amount`, `reimbursable` verdict and sorted `flags` match the
  recomputed truth;
- a line is flagged `unknown_category` when its category is not one of
  `meals`, `lodging`, `transport`, `supplies`; `missing_receipt` when an
  eligible-category line over $25.00 has no receipt; and `over_cap` when an
  otherwise-eligible line would push the running reimbursable total for its
  `(category, date)` above that category's daily cap (caps applied greedily
  in `row_id` order: meals 75, lodging 250, transport 100, supplies 50);
- `flag_counts` reports the number of lines carrying each flag, and
  `policy_compliant` is true only when no line is flagged.

The verifier recomputes all expected values independently (different control
flow from the oracle) from `verifier/data/`; a submission that fabricates
numbers, skips the cap or receipt rules, reorders rows, or tampers with
`/root/expenses.csv` does not pass.
