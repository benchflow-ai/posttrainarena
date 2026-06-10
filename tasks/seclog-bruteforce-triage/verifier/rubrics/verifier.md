# SSH Brute-Force Triage Rubric

Single binary check: the verifier awards 1.0 when `/root/findings.json`
exactly matches the ground truth recomputed from the seeded
`/var/log/auth.log`, and 0.0 otherwise. Concretely, a passing trial:

- exists, is non-empty, and parses as a JSON object with exactly the
  documented keys and JSON types (integers not strings, real booleans,
  `null` for no compromise);
- reports `total_failed_attempts` equal to the count of
  `Failed password for` lines across the whole log (side-effect lines
  such as `Invalid user ...`, `pam_unix(sshd:auth): authentication
  failure`, and disconnect lines are not attempts);
- lists exactly the source IPs with >= 10 failed attempts — no
  below-threshold IPs (e.g. a 9-failure near-miss), no benign users —
  sorted by `failed_attempts` descending, ties by `ip` ascending;
- for each listed IP, matches the recomputed `failed_attempts`,
  `invalid_user_attempts`, sorted `usernames`, `first_failed_line`,
  `last_failed_line`;
- marks an IP `compromised` only when an `Accepted password for` line
  from that IP appears after the IP's 10th failed attempt (a success
  that occurs before the 10th failure does not qualify, and `Accepted
  publickey` never qualifies), with `compromised_user` set to that
  login's username, else `null`;
- sets `any_compromise` to whether any listed IP is compromised.
