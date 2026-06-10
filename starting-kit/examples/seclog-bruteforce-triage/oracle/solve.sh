#!/bin/bash
# Reference solution. Parses /var/log/auth.log per the rules in task.md
# and writes /root/findings.json. Implemented with plain string splitting
# (the verifier independently recomputes the truth with regexes), so an
# agreement between the two is meaningful.
set -e

WORKSPACE="${BENCHFLOW_WORKSPACE:-/root}"
mkdir -p "$WORKSPACE"
export BENCHFLOW_WORKSPACE="$WORKSPACE"

python3 - <<'PYTHON_SCRIPT'
import json
import os

LOG = "/var/log/auth.log"
OUT = os.path.join(os.environ.get("BENCHFLOW_WORKSPACE", "/root"), "findings.json")
THRESHOLD = 10


def parse_user_ip(line, marker):
    """Extract (user, ip, invalid) from '<marker> [invalid user ]U from IP port ...'."""
    rest = line.split(marker, 1)[1]
    invalid = rest.startswith("invalid user ")
    if invalid:
        rest = rest[len("invalid user "):]
    head, _, tail = rest.partition(" from ")
    user = head.strip()
    ip = tail.split()[0]
    return user, ip, invalid


fails = {}    # ip -> list of (lineno, user, invalid)
accepts = {}  # ip -> list of (lineno, user)

with open(LOG, encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        if "Failed password for " in line:
            user, ip, invalid = parse_user_ip(line, "Failed password for ")
            fails.setdefault(ip, []).append((lineno, user, invalid))
        elif "Accepted password for " in line:
            user, ip, _ = parse_user_ip(line, "Accepted password for ")
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
    entries.append({
        "ip": ip,
        "failed_attempts": len(attempts),
        "invalid_user_attempts": sum(1 for _, _, inv in attempts if inv),
        "usernames": sorted({u for _, u, _ in attempts}),
        "first_failed_line": attempts[0][0],
        "last_failed_line": attempts[-1][0],
        "compromised": compromised_user is not None,
        "compromised_user": compromised_user,
    })

entries.sort(key=lambda e: (-e["failed_attempts"], e["ip"]))

findings = {
    "total_failed_attempts": sum(len(v) for v in fails.values()),
    "brute_force_ips": entries,
    "any_compromise": any(e["compromised"] for e in entries),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)
print(f"wrote {OUT}: {len(entries)} brute-force IP(s), "
      f"{findings['total_failed_attempts']} failed attempts, "
      f"any_compromise={findings['any_compromise']}")
PYTHON_SCRIPT

echo "Solution complete."
