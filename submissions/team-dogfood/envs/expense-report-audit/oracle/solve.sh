#!/bin/bash
# Reference solution. Reads /root/expenses.csv, applies the reimbursement
# policy from task.md mechanically, and writes /root/audit.json. Implemented
# with the standard-library csv module and integer-cent arithmetic (the
# verifier independently recomputes the truth), so an agreement between the
# two is meaningful.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"
export BENCHFLOW_WORKSPACE="$WORKSPACE"

python3 - <<'PYTHON_SCRIPT'
import csv
import json
import os

WORKSPACE = os.environ.get("BENCHFLOW_WORKSPACE", "/root")
SRC = os.path.join(WORKSPACE, "expenses.csv")
OUT = os.path.join(WORKSPACE, "audit.json")

ELIGIBLE = {"meals", "lodging", "transport", "supplies"}
# Per (category, date) daily caps, in cents.
CAPS = {"meals": 7500, "lodging": 25000, "transport": 10000, "supplies": 5000}
RECEIPT_THRESHOLD = 2500  # cents; amounts strictly above this need a receipt


def to_cents(s):
    return int(round(float(s) * 100))


rows = []
with open(SRC, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(
            {
                "row_id": int(r["row_id"]),
                "date": r["date"],
                "category": r["category"],
                "amount_cents": to_cents(r["amount"]),
                "receipt": r["receipt"].strip().lower(),
            }
        )

rows.sort(key=lambda r: r["row_id"])

running = {}  # (category, date) -> accepted cents so far
line_items = []
flag_counts = {"missing_receipt": 0, "unknown_category": 0, "over_cap": 0}
total_submitted = 0
total_reimbursable = 0

for r in rows:
    total_submitted += r["amount_cents"]
    flags = []
    reimbursable = False

    if r["category"] not in ELIGIBLE:
        flags.append("unknown_category")
    elif r["amount_cents"] > RECEIPT_THRESHOLD and r["receipt"] != "yes":
        flags.append("missing_receipt")
    else:
        key = (r["category"], r["date"])
        cap = CAPS[r["category"]]
        if running.get(key, 0) + r["amount_cents"] <= cap:
            reimbursable = True
            running[key] = running.get(key, 0) + r["amount_cents"]
        else:
            flags.append("over_cap")

    if reimbursable:
        total_reimbursable += r["amount_cents"]
    for fl in flags:
        flag_counts[fl] += 1

    line_items.append(
        {
            "row_id": r["row_id"],
            "category": r["category"],
            "amount": round(r["amount_cents"] / 100, 2),
            "reimbursable": reimbursable,
            "flags": sorted(flags),
        }
    )

audit = {
    "total_submitted": round(total_submitted / 100, 2),
    "total_reimbursable": round(total_reimbursable / 100, 2),
    "line_items": line_items,
    "flag_counts": flag_counts,
    "policy_compliant": all(not li["flags"] for li in line_items),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
    f.write("\n")
PYTHON_SCRIPT
