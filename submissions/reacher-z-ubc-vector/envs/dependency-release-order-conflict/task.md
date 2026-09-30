---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: software-engineering
  difficulty: hard
  tags: [dependency-release-order, conflict]
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

Construct a valid release sequence from dependency edges. Whenever multiple nodes are ready, choose the lexicographically smallest so the final topological order is uniquely reproducible.

Several rows share a logical key but disagree. Resolve each group by the greatest integer source_priority before doing the domain calculation. Report every resolved logical key in the audit block; retaining both conflicting rows is an error.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
