---
version: '1.0'
metadata:
  author_name: Xiangyi Li
  author_email: xiangyi@benchflow.ai
  difficulty: medium
  category: cybersecurity
  subcategory: log-triage
  category_confidence: high
  task_type:
  - analysis
  - detection
  modality:
  - network-logs
  interface:
  - terminal
  - python
  skill_type:
  - domain-procedure
  - data-cleaning-procedure
  tags:
  - security
  - blue-team
  - ssh
  - auth-log
  - brute-force
  - incident-response
  - log-analysis
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 300
environment:
  build_timeout_sec: 600
  cpus: 1
  memory_mb: 1024
  storage_mb: 4096
  allow_internet: false
---

## prompt

You are a security analyst triaging a possible SSH brute-force incident on
the host `bastion`. The full sshd auth log is at `/var/log/auth.log`. Apply
the threshold rules below mechanically, then write your findings as JSON to
`/root/findings.json`. Do not modify `/var/log/auth.log`.

### Detection rules

1. **Failed attempt** — a log line containing the substring
   `Failed password for`. The source IP is the token after ` from `, and
   the username is the word between `for ` and ` from`, with the
   `invalid user ` prefix stripped when present. Examples:
   - `... sshd[7]: Failed password for root from 198.51.100.23 port 4022 ssh2`
     → user `root`, IP `198.51.100.23`.
   - `... sshd[7]: Failed password for invalid user admin from 203.0.113.45 port 510 ssh2`
     → user `admin`, IP `203.0.113.45`, and it also counts as an
     *invalid-user attempt*.

   Do **not** count `Invalid user ... from ...` preauth lines,
   `pam_unix(sshd:auth): authentication failure` lines, or
   disconnect/connection-closed lines as failed attempts — they are
   side-effects of the same attempt, not extra attempts.
2. **Brute-force IP** — any source IP with **10 or more** failed attempts
   in the whole file.
3. **Successful login** — a log line containing the substring
   `Accepted password for` (same user/IP extraction as rule 1).
   `Accepted publickey` lines do **not** count as successful password
   logins for this analysis.
4. **Compromise** — a brute-force IP is `compromised` if and only if at
   least one of its successful logins appears on a (1-based) line number
   strictly greater than the line number of that IP's **10th** failed
   attempt. If compromised, `compromised_user` is the username of the
   earliest such successful login; otherwise it is `null`.

### Required output — `/root/findings.json`

A single JSON object with exactly these keys:

```json
{
  "total_failed_attempts": 0,
  "brute_force_ips": [
    {
      "ip": "198.51.100.23",
      "failed_attempts": 0,
      "invalid_user_attempts": 0,
      "usernames": ["root"],
      "first_failed_line": 0,
      "last_failed_line": 0,
      "compromised": false,
      "compromised_user": null
    }
  ],
  "any_compromise": false
}
```

Field definitions:

- `total_failed_attempts` (int) — count of failed attempts in the entire
  file from **all** IPs, including those below the brute-force threshold.
- `brute_force_ips` (array) — one object per brute-force IP, sorted by
  `failed_attempts` descending, ties broken by `ip` ascending (plain
  string comparison). For each IP:
  - `ip` (string) — the source IP.
  - `failed_attempts` (int) — its total failed attempts.
  - `invalid_user_attempts` (int) — the subset of its failed attempts
    whose line contains `invalid user`.
  - `usernames` (array of strings) — the distinct usernames targeted in
    its failed attempts, sorted ascending (case-sensitive).
  - `first_failed_line` / `last_failed_line` (int) — 1-based line numbers
    of its first and last failed attempt in the file.
  - `compromised` (bool) and `compromised_user` (string or null) — per
    rule 4.
- `any_compromise` (bool) — true if any brute-force IP is `compromised`.

Use exactly these key names and types. Numbers must be JSON integers (no
strings), and `compromised_user` must be JSON `null` when there is no
qualifying compromise.
