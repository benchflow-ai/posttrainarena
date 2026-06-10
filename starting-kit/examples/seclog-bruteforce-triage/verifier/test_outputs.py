"""Verifier for seclog-bruteforce-triage.

The expected findings are RECOMPUTED here from the verifier's own
pristine copy of the seeded log (verifier/data/auth.log — uploaded with
the verifier after the agent finishes, so the agent can never read or
tamper with it). Nothing in this file hard-codes attacker IPs or counts,
so a fixed-file submission only passes if its content is actually the
correct analysis of the log.

Failure surface:
- missing / empty / non-JSON output  -> TestFindingsFileExists fails
- wrong schema (keys/types)          -> TestSchema fails
- plausible-but-wrong analysis       -> TestTopLevel / TestPerIp fail
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
FINDINGS_FILE = WORKSPACE / "findings.json"
SEED_LOG = Path(__file__).resolve().parent / "data" / "auth.log"

FAIL_RE = re.compile(r"Failed password for (invalid user )?(\S+) from (\S+) port")
ACCEPT_RE = re.compile(r"Accepted password for (\S+) from (\S+) port")
THRESHOLD = 10


def compute_expected() -> dict:
    """Reference implementation of the rules stated in task.md."""
    fails: dict[str, list[tuple[int, str, bool]]] = {}
    accepts: dict[str, list[tuple[int, str]]] = {}
    with open(SEED_LOG, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            m = FAIL_RE.search(line)
            if m:
                invalid, user, ip = bool(m.group(1)), m.group(2), m.group(3)
                fails.setdefault(ip, []).append((lineno, user, invalid))
            m = ACCEPT_RE.search(line)
            if m:
                user, ip = m.group(1), m.group(2)
                accepts.setdefault(ip, []).append((lineno, user))

    entries = []
    for ip, attempts in fails.items():
        if len(attempts) < THRESHOLD:
            continue
        tenth_line = attempts[THRESHOLD - 1][0]
        compromised_user = None
        for acc_line, acc_user in accepts.get(ip, []):
            if acc_line > tenth_line:
                compromised_user = acc_user
                break
        entries.append(
            {
                "ip": ip,
                "failed_attempts": len(attempts),
                "invalid_user_attempts": sum(1 for _, _, inv in attempts if inv),
                "usernames": sorted({user for _, user, _ in attempts}),
                "first_failed_line": attempts[0][0],
                "last_failed_line": attempts[-1][0],
                "compromised": compromised_user is not None,
                "compromised_user": compromised_user,
            }
        )
    entries.sort(key=lambda e: (-e["failed_attempts"], e["ip"]))
    return {
        "total_failed_attempts": sum(len(v) for v in fails.values()),
        "brute_force_ips": entries,
        "any_compromise": any(e["compromised"] for e in entries),
    }


@pytest.fixture(scope="module")
def expected() -> dict:
    assert SEED_LOG.exists(), f"verifier seed log missing at {SEED_LOG}"
    exp = compute_expected()
    # Internal consistency guards on the recomputed truth itself —
    # if the seed data ever drifts, fail loudly instead of grading
    # against a degenerate expectation.
    assert exp["total_failed_attempts"] > 0
    assert len(exp["brute_force_ips"]) >= 2
    return exp


@pytest.fixture(scope="module")
def submitted() -> dict:
    assert FINDINGS_FILE.exists(), f"Findings file not found at {FINDINGS_FILE}"
    raw = FINDINGS_FILE.read_text(encoding="utf-8").strip()
    assert raw, f"{FINDINGS_FILE} is empty"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        pytest.fail(f"{FINDINGS_FILE} is not valid JSON: {e}")
    assert isinstance(data, dict), "findings.json must be a JSON object"
    return data


class TestFindingsFileExists:
    """(a) missing / empty / malformed output fails here."""

    def test_file_exists_and_parses(self, submitted):
        assert isinstance(submitted, dict)


class TestSchema:
    """Exact key set and JSON types from the documented schema."""

    TOP_KEYS = {"total_failed_attempts", "brute_force_ips", "any_compromise"}
    ENTRY_KEYS = {
        "ip",
        "failed_attempts",
        "invalid_user_attempts",
        "usernames",
        "first_failed_line",
        "last_failed_line",
        "compromised",
        "compromised_user",
    }

    def test_top_level_keys(self, submitted):
        assert set(submitted.keys()) == self.TOP_KEYS, (
            f"Top-level keys must be exactly {sorted(self.TOP_KEYS)}, "
            f"got {sorted(submitted.keys())}"
        )

    def test_top_level_types(self, submitted):
        assert isinstance(submitted["total_failed_attempts"], int) and not isinstance(
            submitted["total_failed_attempts"], bool
        ), "total_failed_attempts must be a JSON integer"
        assert isinstance(submitted["brute_force_ips"], list), "brute_force_ips must be an array"
        assert isinstance(submitted["any_compromise"], bool), "any_compromise must be a boolean"

    def test_entry_keys_and_types(self, submitted):
        problems = []
        for i, entry in enumerate(submitted["brute_force_ips"]):
            if not isinstance(entry, dict):
                problems.append(f"entry {i}: not an object")
                continue
            if set(entry.keys()) != self.ENTRY_KEYS:
                problems.append(
                    f"entry {i}: keys must be exactly {sorted(self.ENTRY_KEYS)}, "
                    f"got {sorted(entry.keys())}"
                )
                continue
            for k in ("failed_attempts", "invalid_user_attempts", "first_failed_line", "last_failed_line"):
                if not isinstance(entry[k], int) or isinstance(entry[k], bool):
                    problems.append(f"entry {i}: {k} must be a JSON integer, got {entry[k]!r}")
            if not isinstance(entry["ip"], str):
                problems.append(f"entry {i}: ip must be a string")
            if not isinstance(entry["usernames"], list) or not all(
                isinstance(u, str) for u in entry["usernames"]
            ):
                problems.append(f"entry {i}: usernames must be an array of strings")
            if not isinstance(entry["compromised"], bool):
                problems.append(f"entry {i}: compromised must be a boolean")
            if entry["compromised_user"] is not None and not isinstance(entry["compromised_user"], str):
                problems.append(f"entry {i}: compromised_user must be a string or null")
        assert not problems, "Schema problems:\n" + "\n".join(problems)


class TestTopLevel:
    """(b) plausible-but-wrong totals / verdicts fail here."""

    def test_total_failed_attempts(self, submitted, expected):
        assert submitted["total_failed_attempts"] == expected["total_failed_attempts"], (
            f"total_failed_attempts: expected {expected['total_failed_attempts']}, "
            f"got {submitted['total_failed_attempts']}"
        )

    def test_brute_force_ip_set(self, submitted, expected):
        got = [e.get("ip") for e in submitted["brute_force_ips"] if isinstance(e, dict)]
        want = [e["ip"] for e in expected["brute_force_ips"]]
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        assert not missing and not extra, (
            f"Brute-force IP set mismatch. Missing: {missing or 'none'}; "
            f"unexpected extras (check the >=10 threshold): {extra or 'none'}"
        )

    def test_ordering(self, submitted, expected):
        got = [e.get("ip") for e in submitted["brute_force_ips"] if isinstance(e, dict)]
        want = [e["ip"] for e in expected["brute_force_ips"]]
        if set(got) == set(want):
            assert got == want, (
                "brute_force_ips must be sorted by failed_attempts descending, "
                f"ties by ip ascending. Expected order {want}, got {got}"
            )

    def test_any_compromise(self, submitted, expected):
        assert submitted["any_compromise"] == expected["any_compromise"], (
            f"any_compromise: expected {expected['any_compromise']}, "
            f"got {submitted['any_compromise']}"
        )


class TestPerIp:
    """Every per-IP field must match the recomputed ground truth."""

    def test_per_ip_fields(self, submitted, expected):
        got_by_ip = {
            e["ip"]: e
            for e in submitted["brute_force_ips"]
            if isinstance(e, dict) and isinstance(e.get("ip"), str)
        }
        errors = []
        for exp_entry in expected["brute_force_ips"]:
            ip = exp_entry["ip"]
            got = got_by_ip.get(ip)
            if got is None:
                errors.append(f"{ip}: missing from brute_force_ips")
                continue
            for key, want in exp_entry.items():
                if got.get(key) != want:
                    errors.append(f"{ip}.{key}: expected {want!r}, got {got.get(key)!r}")
        assert not errors, "Per-IP mismatches:\n" + "\n".join(errors)
