---
version: '1.0'
metadata:
  author_name: Xiangyi Li
  author_email: xiangyi@benchflow.ai
  difficulty: medium
  category: office-white-collar
  subcategory: expense-policy-audit
  category_confidence: high
  task_type:
  - analysis
  - calculation
  modality:
  - csv
  interface:
  - terminal
  - python
  skill_type:
  - domain-procedure
  - data-cleaning-procedure
  tags:
  - finance-operations
  - expense-report
  - policy-compliance
  - reimbursement
  - csv
  - back-office
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

You are an accounts-payable analyst auditing a corporate travel expense
report against the reimbursement policy. The submitted line items are in a
CSV at `/root/expenses.csv`. Apply the policy rules below mechanically, then
write your audit as JSON to `/root/audit.json`. Do not modify
`/root/expenses.csv`.

### Input — `/root/expenses.csv`

A header row followed by one row per expense line. The columns are:

```
row_id,date,category,merchant,amount,receipt
```

- `row_id` (int) — a stable 1-based identifier, unique and already in
  ascending order.
- `date` (string) — the expense date, `YYYY-MM-DD`.
- `category` (string) — the submitter's category label (free text).
- `merchant` (string) — where the charge was incurred.
- `amount` (number) — the charge in US dollars, always positive with two
  decimal places.
- `receipt` (string) — `yes` if a receipt was attached, otherwise `no`.

For example, a row such as `7,2026-04-06,meals,SteakHouse,61.00,yes` is the
seventh line item: a $61.00 meals charge with a receipt.

### Policy (apply in this order, per line)

Evaluate each line independently and in `row_id` order. A line is
**reimbursable** only if it passes every applicable rule; the first rule it
fails assigns its single flag and makes it non-reimbursable.

1. **Eligible category.** Only `meals`, `lodging`, `transport`, and
   `supplies` are reimbursable categories. If `category` is anything else,
   flag the line `unknown_category` and stop (it is not reimbursable, and no
   further rule applies to it).
2. **Receipt requirement.** For an eligible-category line whose `amount` is
   strictly greater than **$25.00**, a receipt is required. If `receipt` is
   not `yes`, flag the line `missing_receipt` and stop. (Lines at or below
   $25.00 do not require a receipt.)
3. **Daily category cap.** Each eligible category has a per-day cap:
   **meals $75.00, lodging $250.00, transport $100.00, supplies $50.00**.
   The cap applies to the running total of *accepted* (reimbursable) amounts
   within the same `(category, date)`. Walking the lines in `row_id` order,
   a candidate line that has passed rules 1–2 is **accepted** if adding its
   `amount` to the already-accepted total for its `(category, date)` keeps
   that total at or below the cap; then its amount counts toward that running
   total. Otherwise flag it `over_cap` and leave it non-reimbursable — and do
   **not** add it to the running total (a rejected line never consumes cap,
   so a later, smaller line on the same day can still be accepted).

A line that passes all three rules is reimbursable with an empty flag list.
Every line receives at most one flag.

### Required output — `/root/audit.json`

A single JSON object with exactly these keys:

```json
{
  "total_submitted": 0.00,
  "total_reimbursable": 0.00,
  "line_items": [
    {
      "row_id": 0,
      "category": "<the row's category, copied verbatim>",
      "amount": 0.00,
      "reimbursable": false,
      "flags": ["<zero or one of the flag names above>"]
    }
  ],
  "flag_counts": {"missing_receipt": 0, "unknown_category": 0, "over_cap": 0},
  "policy_compliant": false
}
```

Field definitions:

- `total_submitted` (number) — the sum of `amount` across **every** line in
  the file, rounded to cents.
- `total_reimbursable` (number) — the sum of `amount` across only the
  reimbursable lines, rounded to cents.
- `line_items` (array) — one object per CSV row, in `row_id` ascending order.
  For each line: `row_id` (int), `category` (the verbatim label), `amount`
  (number, rounded to cents), `reimbursable` (bool), and `flags` (array of
  strings — empty when reimbursable, otherwise the single applicable flag).
- `flag_counts` (object) — the number of lines carrying each flag, with
  exactly the keys `missing_receipt`, `unknown_category`, `over_cap`
  (integers, zero when a flag does not occur).
- `policy_compliant` (bool) — `true` only if no line carries any flag.

Use exactly these key names and types. Money values are JSON numbers rounded
to two decimals (trailing zeros need not be preserved — `42` and `42.00` are
equal), counts are JSON integers, and booleans are real JSON booleans.
