---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: industrial-and-energy-operations
  difficulty: hard
  tags: [sla-escalation, recovery]
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  build_timeout_sec: 600
  cpus: 1
  memory_mb: 2048
  storage_mb: 2048
  allow_internet: false
---

## prompt

Compare each service case deadline with the frozen analysis timestamp. Classify it as breached, due within twenty-four hours, or on-time without consulting wall-clock time.

A prior attempt is supplied in prior_answer and is intentionally incomplete. Recompute the result from source records rather than trusting it, then list the top-level prior fields that required correction in the audit block.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
