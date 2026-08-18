---
version: "1.0"
metadata:
  author_name: Yuxuan Zhang
  author_email: reacher@cs.ubc.ca
  category: ai-ml-and-agentic-systems
  difficulty: hard
  tags: [entity-survivorship, temporal]
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

Resolve repeated entity observations through recorded-time survivorship. For every entity retain only the latest authoritative value and emit a deterministic entity-to-value mapping.

The fixture is an as-of analysis. Exclude records whose effective_at is later than the declared as_of timestamp before computing the domain result, and list the excluded future record IDs in the audit block.

Read `/root/instance.json` and write `/root/answer.json`. Preserve deterministic key ordering, do not modify the input, and include the required `audit` object alongside the domain result.
