---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: industrial-and-energy-operations
  difficulty: hard
  tags: [sla-escalation, baseline]
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

Treat every declared record as authoritative. Produce the primary result plus an audit block reporting the exact number of records used. This case establishes the ordinary workflow contract.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
