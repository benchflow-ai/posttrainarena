---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: natural-science
  difficulty: hard
  tags: [scientific-robust-summary, adversarial]
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

Compute a symmetric ten-percent trimmed mean for every sensor series after ordering its observations. Report each robust center and the largest cross-sensor spread to two decimals.

Rows marked is_distractor=true are realistic decoys and must not influence the result. Exclude them, report their record IDs in the audit block, and use every remaining row exactly once.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
