"""Verifier for expense-report-audit.

The expected audit is RECOMPUTED here from the verifier's own pristine copy
of the seeded expense CSV (verifier/data/expenses.csv — uploaded with the
verifier after the agent finishes, so the agent can neither read it during
the run nor tamper with the ground truth). Nothing in this file hard-codes a
total, a flag count, or a per-row verdict, so a fixed-file submission only
passes if its content is the correct analysis of the CSV.

The recompute below is written independently of oracle/solve.sh (different
control flow), so an agreement between the two is meaningful rather than a
copy. Money is handled in integer cents end-to-end to avoid float drift, and
the few float comparisons that remain use an explicit tolerance.

Failure surface:
- missing / empty / non-JSON output  -> TestAuditFileExists fails
- wrong schema (keys/types)          -> TestSchema fails
- plausible-but-wrong analysis       -> TestTotals / TestLineItems fail
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
AUDIT_FILE = WORKSPACE / "audit.json"
SEED_CSV = Path(__file__).resolve().parent / "data" / "expenses.csv"

ELIGIBLE = {"meals", "lodging", "transport", "supplies"}
CAPS_CENTS = {"meals": 7500, "lodging": 25000, "transport": 10000, "supplies": 5000}
RECEIPT_THRESHOLD_CENTS = 2500
MONEY_TOL = 0.005  # half a cent


def _cents(s: str) -> int:
    return int(round(float(s) * 100))


def compute_expected() -> dict:
    """Reference implementation of the policy stated in task.md, recomputed
    from the verifier-private seed CSV."""
    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        raw = sorted(csv.DictReader(f), key=lambda r: int(r["row_id"]))

    accepted: dict[tuple[str, str], int] = {}
    items = []
    counts = {"missing_receipt": 0, "unknown_category": 0, "over_cap": 0}
    submitted = 0
    reimbursable_total = 0

    for r in raw:
        amount_c = _cents(r["amount"])
        submitted += amount_c
        category = r["category"]
        flags: list[str] = []
        ok = False

        if category not in ELIGIBLE:
            flags = ["unknown_category"]
        elif amount_c > RECEIPT_THRESHOLD_CENTS and r["receipt"].strip().lower() != "yes":
            flags = ["missing_receipt"]
        else:
            key = (category, r["date"])
            if accepted.get(key, 0) + amount_c <= CAPS_CENTS[category]:
                ok = True
                accepted[key] = accepted.get(key, 0) + amount_c
            else:
                flags = ["over_cap"]

        if ok:
            reimbursable_total += amount_c
        for fl in flags:
            counts[fl] += 1

        items.append(
            {
                "row_id": int(r["row_id"]),
                "category": category,
                "amount": round(amount_c / 100, 2),
                "reimbursable": ok,
                "flags": flags,
            }
        )

    return {
        "total_submitted": round(submitted / 100, 2),
        "total_reimbursable": round(reimbursable_total / 100, 2),
        "line_items": items,
        "flag_counts": counts,
        "policy_compliant": all(not it["flags"] for it in items),
    }


@pytest.fixture(scope="module")
def expected() -> dict:
    assert SEED_CSV.exists(), f"verifier seed CSV missing at {SEED_CSV}"
    exp = compute_expected()
    # Internal consistency guards on the recomputed truth itself — if the seed
    # data ever drifts to something degenerate, fail loudly instead of grading
    # against a trivial expectation.
    assert len(exp["line_items"]) >= 10, "seed CSV unexpectedly small"
    assert exp["total_submitted"] > 0
    assert sum(exp["flag_counts"].values()) >= 3, "seed should exercise every flag"
    assert all(exp["flag_counts"][k] >= 1 for k in exp["flag_counts"]), (
        "each flag type should appear at least once in the seed"
    )
    assert not exp["policy_compliant"], "seed should contain at least one violation"
    return exp


@pytest.fixture(scope="module")
def submitted() -> dict:
    assert AUDIT_FILE.exists(), f"Audit file not found at {AUDIT_FILE}"
    raw = AUDIT_FILE.read_text(encoding="utf-8").strip()
    assert raw, f"{AUDIT_FILE} is empty"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        pytest.fail(f"{AUDIT_FILE} is not valid JSON: {e}")
    assert isinstance(data, dict), "audit.json must be a JSON object"
    return data


class TestAuditFileExists:
    """(a) missing / empty / malformed output fails here."""

    def test_file_exists_and_parses(self, submitted):
        assert isinstance(submitted, dict)


class TestSchema:
    """Exact key set and JSON types from the documented schema."""

    TOP_KEYS = {
        "total_submitted",
        "total_reimbursable",
        "line_items",
        "flag_counts",
        "policy_compliant",
    }
    ITEM_KEYS = {"row_id", "category", "amount", "reimbursable", "flags"}
    COUNT_KEYS = {"missing_receipt", "unknown_category", "over_cap"}

    def _is_number(self, x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def test_top_level_keys(self, submitted):
        assert set(submitted.keys()) == self.TOP_KEYS, (
            f"Top-level keys must be exactly {sorted(self.TOP_KEYS)}, "
            f"got {sorted(submitted.keys())}"
        )

    def test_top_level_types(self, submitted):
        assert self._is_number(submitted["total_submitted"]), "total_submitted must be a number"
        assert self._is_number(submitted["total_reimbursable"]), "total_reimbursable must be a number"
        assert isinstance(submitted["line_items"], list), "line_items must be an array"
        assert isinstance(submitted["flag_counts"], dict), "flag_counts must be an object"
        assert isinstance(submitted["policy_compliant"], bool), "policy_compliant must be a boolean"

    def test_flag_counts_schema(self, submitted):
        fc = submitted["flag_counts"]
        assert set(fc.keys()) == self.COUNT_KEYS, (
            f"flag_counts keys must be exactly {sorted(self.COUNT_KEYS)}, got {sorted(fc.keys())}"
        )
        for k, v in fc.items():
            assert isinstance(v, int) and not isinstance(v, bool), f"flag_counts.{k} must be an integer"

    def test_item_keys_and_types(self, submitted):
        problems = []
        for i, item in enumerate(submitted["line_items"]):
            if not isinstance(item, dict):
                problems.append(f"item {i}: not an object")
                continue
            if set(item.keys()) != self.ITEM_KEYS:
                problems.append(
                    f"item {i}: keys must be exactly {sorted(self.ITEM_KEYS)}, got {sorted(item.keys())}"
                )
                continue
            if not isinstance(item["row_id"], int) or isinstance(item["row_id"], bool):
                problems.append(f"item {i}: row_id must be an integer")
            if not isinstance(item["category"], str):
                problems.append(f"item {i}: category must be a string")
            if not self._is_number(item["amount"]):
                problems.append(f"item {i}: amount must be a number")
            if not isinstance(item["reimbursable"], bool):
                problems.append(f"item {i}: reimbursable must be a boolean")
            if not isinstance(item["flags"], list) or not all(isinstance(x, str) for x in item["flags"]):
                problems.append(f"item {i}: flags must be an array of strings")
        assert not problems, "Schema problems:\n" + "\n".join(problems)


class TestTotals:
    """(b) plausible-but-wrong totals / verdicts fail here."""

    def test_total_submitted(self, submitted, expected):
        assert abs(submitted["total_submitted"] - expected["total_submitted"]) <= MONEY_TOL, (
            f"total_submitted: expected {expected['total_submitted']}, got {submitted['total_submitted']}"
        )

    def test_total_reimbursable(self, submitted, expected):
        assert abs(submitted["total_reimbursable"] - expected["total_reimbursable"]) <= MONEY_TOL, (
            f"total_reimbursable: expected {expected['total_reimbursable']}, "
            f"got {submitted['total_reimbursable']}"
        )

    def test_flag_counts(self, submitted, expected):
        assert submitted["flag_counts"] == expected["flag_counts"], (
            f"flag_counts: expected {expected['flag_counts']}, got {submitted['flag_counts']}"
        )

    def test_policy_compliant(self, submitted, expected):
        assert submitted["policy_compliant"] == expected["policy_compliant"], (
            f"policy_compliant: expected {expected['policy_compliant']}, got {submitted['policy_compliant']}"
        )


class TestLineItems:
    """Every per-row field must match the recomputed ground truth."""

    def test_row_id_set_and_order(self, submitted, expected):
        got = [it.get("row_id") for it in submitted["line_items"] if isinstance(it, dict)]
        want = [it["row_id"] for it in expected["line_items"]]
        assert sorted(got) == sorted(want), (
            f"row_id set mismatch. Missing: {sorted(set(want) - set(got))}; "
            f"unexpected: {sorted(set(got) - set(want))}"
        )
        assert got == want, (
            f"line_items must be sorted by row_id ascending. Expected {want}, got {got}"
        )

    def test_per_row_fields(self, submitted, expected):
        got_by_id = {
            it["row_id"]: it
            for it in submitted["line_items"]
            if isinstance(it, dict) and isinstance(it.get("row_id"), int)
        }
        errors = []
        for exp_item in expected["line_items"]:
            rid = exp_item["row_id"]
            got = got_by_id.get(rid)
            if got is None:
                errors.append(f"row {rid}: missing")
                continue
            if got.get("category") != exp_item["category"]:
                errors.append(f"row {rid}.category: expected {exp_item['category']!r}, got {got.get('category')!r}")
            if not isinstance(got.get("amount"), (int, float)) or abs(got["amount"] - exp_item["amount"]) > MONEY_TOL:
                errors.append(f"row {rid}.amount: expected {exp_item['amount']}, got {got.get('amount')}")
            if got.get("reimbursable") != exp_item["reimbursable"]:
                errors.append(f"row {rid}.reimbursable: expected {exp_item['reimbursable']}, got {got.get('reimbursable')}")
            if sorted(got.get("flags", [])) != sorted(exp_item["flags"]):
                errors.append(f"row {rid}.flags: expected {sorted(exp_item['flags'])}, got {got.get('flags')}")
        assert not errors, "Per-row mismatches:\n" + "\n".join(errors)
