---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: media-and-multimodal-content
  difficulty: hard
  tags: [media-timeline-qc, recovery]
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

Inspect ordered media segments for temporal collisions. Identify each adjacent overlap with its duration and calculate the exact aggregate overlap across the delivered timeline.

A prior attempt is supplied in prior_answer and is intentionally incomplete. Recompute the result from source records rather than trusting it, then list the top-level prior fields that required correction in the audit block.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
