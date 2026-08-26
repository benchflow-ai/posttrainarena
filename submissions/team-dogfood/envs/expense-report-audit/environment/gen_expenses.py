#!/usr/bin/env python3
"""Deterministic generator that authored seed/expenses.csv.

Kept for provenance and so the dataset can be regenerated or extended; it is
intentionally NOT copied into the environment image (the Dockerfile copies
only seed/expenses.csv). The committed CSV is the authoritative artifact —
running this script reproduces it byte-for-byte.

The rows are designed to exercise every branch of the policy at least once:
clean reimbursable lines, unknown categories, missing receipts on
over-threshold amounts, and order-dependent per-day cap rejections across
several (category, date) pairs.

    python3 gen_expenses.py > expenses.csv
"""
from __future__ import annotations

import csv
import sys

ROWS = [
    # date,        category,      merchant,        amount,  receipt
    ("2026-04-06", "transport",   "CityCab",        "42.00", "yes"),
    ("2026-04-06", "meals",       "DinerOne",       "18.50", "yes"),
    ("2026-04-06", "meals",       "CoffeeBar",       "9.75", "no"),
    ("2026-04-06", "lodging",     "GrandStay",     "210.00", "yes"),
    ("2026-04-06", "supplies",    "OfficeMax",      "12.30", "yes"),
    ("2026-04-06", "entertainment", "MovieHouse",   "30.00", "yes"),
    ("2026-04-06", "meals",       "SteakHouse",     "61.00", "yes"),
    ("2026-04-07", "transport",   "Metro",           "3.50", "no"),
    ("2026-04-07", "transport",   "AirportShuttle", "55.00", "no"),
    ("2026-04-07", "meals",       "BistroX",        "40.00", "yes"),
    ("2026-04-07", "meals",       "Cafe88",         "28.00", "yes"),
    ("2026-04-07", "meals",       "NightGrill",     "22.00", "yes"),
    ("2026-04-07", "lodging",     "GrandStay",     "210.00", "yes"),
    ("2026-04-07", "supplies",    "PaperCo",        "45.00", "yes"),
    ("2026-04-07", "supplies",    "InkJet",          "9.99", "yes"),
    ("2026-04-07", "gift",        "GiftShop",       "15.00", "yes"),
    ("2026-04-08", "transport",   "CityCab",        "48.00", "yes"),
    ("2026-04-08", "transport",   "CityCab",        "60.00", "yes"),
    ("2026-04-08", "meals",       "BreakfastNook",  "12.00", "yes"),
    ("2026-04-08", "meals",       "LunchSpot",      "26.00", "no"),
    ("2026-04-08", "meals",       "DinnerClub",     "30.00", "yes"),
    ("2026-04-08", "lodging",     "BudgetInn",      "80.00", "no"),
    ("2026-04-08", "supplies",    "OfficeMax",      "20.00", "yes"),
    ("2026-04-08", "training",    "Seminar",       "500.00", "yes"),
    ("2026-04-08", "meals",       "SnackBar",        "5.00", "no"),
]


def main() -> int:
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["row_id", "date", "category", "merchant", "amount", "receipt"])
    for i, (date, category, merchant, amount, receipt) in enumerate(ROWS, start=1):
        w.writerow([i, date, category, merchant, amount, receipt])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
