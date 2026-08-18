---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: industrial-and-energy-operations
  difficulty: hard
  tags: [route-distance-audit, adversarial]
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

Recompute the declared route one directed leg at a time. Sum available edge distances to two decimals and enumerate every consecutive leg absent from the network table.

Rows marked is_distractor=true are realistic decoys and must not influence the result. Exclude them, report their record IDs in the audit block, and use every remaining row exactly once.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
