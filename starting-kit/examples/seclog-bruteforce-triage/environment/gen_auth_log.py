#!/usr/bin/env python3
"""Deterministic generator for the seeded /var/log/auth.log.

Run from this directory:

    python3 gen_auth_log.py

Writes `seed/auth.log` (the file the Dockerfile copies to
/var/log/auth.log) and prints summary statistics. Everything is driven
by a fixed-seed random.Random, so the output is byte-identical across
runs and machines. The byte-identical copy in ../verifier/data/auth.log
must be refreshed whenever this file changes:

    python3 gen_auth_log.py && cp seed/auth.log ../verifier/data/auth.log

NOTE: this script is NOT copied into the task image — the agent only
ever sees the rendered log. The script intentionally lives outside
environment/Dockerfile COPY lines so the actor table below never leaks
into the sandbox.
"""

from __future__ import annotations

import random
from pathlib import Path

SEED = 1337
HOST = "bastion"
OUT = Path(__file__).parent / "seed" / "auth.log"

INVALID_USERS = [
    "admin", "test", "oracle", "postgres", "ftpuser", "ubuntu", "pi",
    "git", "jenkins", "mysql", "nagios", "www", "user", "guest",
    "support", "dev", "hadoop", "tomcat", "vagrant", "minecraft",
]
VALID_USERS = {"root", "alice", "bob", "carol", "deploy"}


def fail_event(rng: random.Random, ip: str, user: str) -> list[str]:
    """One failed SSH auth attempt: exactly one 'Failed password for'
    line plus 0-3 realistic companion lines sharing the same port."""
    invalid = user not in VALID_USERS
    port = rng.randint(1024, 65000)
    lines: list[str] = []
    if invalid and rng.random() < 0.7:
        lines.append(f"Invalid user {user} from {ip} port {port}")
    if rng.random() < 0.5:
        suffix = "" if invalid else f" user={user}"
        lines.append(
            "pam_unix(sshd:auth): authentication failure; logname= uid=0 "
            f"euid=0 tty=ssh ruser= rhost={ip}{suffix}"
        )
    who = f"invalid user {user}" if invalid else user
    lines.append(f"Failed password for {who} from {ip} port {port} ssh2")
    tail = rng.random()
    label = f"invalid user {user}" if invalid else f"authenticating user {user}"
    if tail < 0.35:
        lines.append(f"Connection closed by {label} {ip} port {port} [preauth]")
    elif tail < 0.6:
        lines.append(f"Received disconnect from {ip} port {port}:11: Bye Bye [preauth]")
        lines.append(f"Disconnected from {label} {ip} port {port} [preauth]")
    return lines


def accept_event(rng: random.Random, ip: str, user: str, method: str = "password") -> list[str]:
    port = rng.randint(1024, 65000)
    uid = 0 if user == "root" else rng.choice([1000, 1001, 1002, 1003])
    lines = [
        f"Accepted {method} for {user} from {ip} port {port} ssh2",
        f"pam_unix(sshd:session): session opened for user {user}(uid={uid}) by (uid=0)",
    ]
    if rng.random() < 0.6:
        lines.append(f"pam_unix(sshd:session): session closed for user {user}")
    return lines


def scanner_event(rng: random.Random, ip: str) -> list[str]:
    port = rng.randint(1024, 65000)
    kind = rng.random()
    if kind < 0.4:
        return [f"Did not receive identification string from {ip} port {port}"]
    if kind < 0.7:
        return [f"Connection closed by {ip} port {port} [preauth]"]
    return [
        f"Connection reset by {ip} port {port} [preauth]",
    ]


def cron_event(rng: random.Random) -> list[str]:
    return [
        "CRON:pam_unix(cron:session): session opened for user root(uid=0) by (uid=0)",
        "CRON:pam_unix(cron:session): session closed for user root",
    ]


def sudo_event(rng: random.Random) -> list[str]:
    user = rng.choice(["alice", "deploy"])
    cmd = rng.choice(["/usr/bin/systemctl restart nginx", "/usr/bin/apt-get update", "/usr/bin/journalctl -u sshd"])
    return [
        f"sudo:    {user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND={cmd}",
        f"sudo:pam_unix(sudo:session): session opened for user root(uid=0) by {user}(uid=1000)",
        f"sudo:pam_unix(sudo:session): session closed for user root",
    ]


def build_queues(rng: random.Random) -> dict[str, list[list[str]]]:
    queues: dict[str, list[list[str]]] = {}

    # Attacker A — loud dictionary attack, invalid users only, never succeeds.
    a_ip = "203.0.113.45"
    queues["atk_a"] = [
        fail_event(rng, a_ip, rng.choice(INVALID_USERS)) for _ in range(850)
    ]

    # Attacker B — persistent root brute force, SUCCEEDS after all 120 failures.
    b_ip = "198.51.100.23"
    q = [fail_event(rng, b_ip, "root") for _ in range(120)]
    q.append(accept_event(rng, b_ip, "root", "password"))
    queues["atk_b"] = q

    # Attacker C — admin-flavoured wordlist, mixed valid/invalid targets.
    c_ip = "192.0.2.146"
    queues["atk_c"] = [
        fail_event(rng, c_ip, rng.choice(["admin", "administrator", "root"]))
        for _ in range(75)
    ]

    # Attacker E — targets the valid 'deploy' account; guesses the password
    # on what is only its 7th event (6 failures first), then keeps failing.
    # Under the stated rule (success AFTER the 10th failure) this is NOT a
    # qualifying compromise.
    e_ip = "192.0.2.200"
    q = [fail_event(rng, e_ip, "deploy") for _ in range(6)]
    q.append(accept_event(rng, e_ip, "deploy", "password"))
    q.extend(fail_event(rng, e_ip, "deploy") for _ in range(34))
    queues["atk_e"] = q

    # Attacker D — small burst, just over the threshold.
    d_ip = "203.0.113.77"
    queues["atk_d"] = [
        fail_event(rng, d_ip, rng.choice(["root", "ubuntu"])) for _ in range(15)
    ]

    # Near-miss — 9 failures, one below the threshold. Must NOT be flagged.
    n_ip = "198.51.100.9"
    queues["near"] = [fail_event(rng, n_ip, "root") for _ in range(9)]

    # Benign users.
    alice = [accept_event(rng, "10.0.0.5", "alice", "password") for _ in range(25)]
    alice.insert(4, fail_event(rng, "10.0.0.5", "alice"))
    alice.insert(15, fail_event(rng, "10.0.0.5", "alice"))
    queues["alice"] = alice

    bob = [accept_event(rng, "10.0.0.12", "bob", "publickey") for _ in range(18)]
    bob.insert(7, fail_event(rng, "10.0.0.12", "bob"))
    queues["bob"] = bob

    queues["deploy_ci"] = [
        accept_event(rng, "10.0.0.7", "deploy", "publickey") for _ in range(30)
    ]

    # Carol forgot her password: 3 failures, then in.
    carol = [fail_event(rng, "10.0.0.21", "carol") for _ in range(3)]
    carol.append(accept_event(rng, "10.0.0.21", "carol", "password"))
    queues["carol"] = carol

    # Background scanners that never get as far as an auth attempt.
    scanner_ips = ["185.220.101.34", "141.98.10.60", "45.155.205.99", "89.248.165.74"]
    queues["scanners"] = [
        scanner_event(rng, rng.choice(scanner_ips)) for _ in range(80)
    ]

    queues["cron"] = [cron_event(rng) for _ in range(40)]
    queues["sudo"] = [sudo_event(rng) for _ in range(15)]
    return queues


def render(rng: random.Random, queues: dict[str, list[list[str]]]) -> list[str]:
    """Interleave per-actor event queues (preserving each queue's order)
    onto a monotonically increasing June timeline."""
    out: list[str] = []
    t = 0  # seconds since Jun 1 00:00:00
    names = list(queues)
    while any(queues[n] for n in names):
        live = [n for n in names if queues[n]]
        weights = [len(queues[n]) for n in live]
        actor = rng.choices(live, weights=weights, k=1)[0]
        event = queues[actor].pop(0)
        pid = rng.randint(800, 99999)
        for body in event:
            t += rng.randint(0, 2)
            day, rem = divmod(t, 86400)
            hh, rem = divmod(rem, 3600)
            mm, ss = divmod(rem, 60)
            ts = f"Jun {day + 1:>2} {hh:02}:{mm:02}:{ss:02}"
            if body.startswith("CRON:"):
                out.append(f"{ts} {HOST} CRON[{pid}]: {body[5:]}")
            elif body.startswith("sudo:"):
                out.append(f"{ts} {HOST} sudo[{pid}]: {body[5:]}")
            else:
                out.append(f"{ts} {HOST} sshd[{pid}]: {body}")
        t += rng.randint(1, 20)
    return out


def main() -> None:
    rng = random.Random(SEED)
    queues = build_queues(rng)
    lines = render(rng, queues)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [ln for ln in lines if "Failed password for" in ln]
    accepted = [ln for ln in lines if "Accepted password for" in ln]
    print(f"wrote {OUT} ({len(lines)} lines)")
    print(f"  Failed password lines:   {len(failed)}")
    print(f"  Accepted password lines: {len(accepted)}")


if __name__ == "__main__":
    main()
